#include "serve/cluster_config.hpp"

#include <arpa/inet.h>

#include <cmath>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/bf16_residency.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/minijson.hpp"

namespace dgpp::serve {

namespace {

using dgpp::minijson::Member;
using dgpp::minijson::Value;

[[noreturn]] void fail(const std::string& what, const std::string& msg) {
  throw std::runtime_error("cluster config " + what + ": " + msg);
}

int64_t integer(const Value& v, const std::string& key, const std::string& what,
                int64_t lo, int64_t hi) {
  if (!v.is_number() || v.as_double(0.0) != static_cast<double>(v.as_int(0)))
    fail(what, "'" + key + "' must be an integer");
  const int64_t x = v.as_int(0);
  if (x < lo || x > hi)
    fail(what, "'" + key + "' must be in [" + std::to_string(lo) + ", " +
                   std::to_string(hi) + "]");
  return x;
}

double number(const Value& v, const std::string& key, const std::string& what) {
  if (!v.is_number() || !std::isfinite(v.as_double(0.0)))
    fail(what, "'" + key + "' must be a finite number");
  return v.as_double(0.0);
}

bool boolean(const Value& v, const std::string& key, const std::string& what) {
  if (!v.is_bool()) fail(what, "'" + key + "' must be true or false");
  return v.as_bool(false);
}

std::string text(const Value& v, const std::string& key, const std::string& what) {
  if (!v.is_string()) fail(what, "'" + key + "' must be a string");
  return std::string(v.as_string());
}

}  // namespace

std::string expand_home(const std::string& path) {
  if (path.empty() || path[0] != '~') return path;
  if (path.size() > 1 && path[1] != '/') return path;
  const char* home = std::getenv("HOME");
  if (home == nullptr || !*home) return path;
  return std::string(home) + path.substr(1);
}

std::string config_digest(const std::string& canonical) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (const unsigned char c : canonical) {
    h ^= c;
    h *= 0x100000001b3ull;
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

ClusterConfig parse_cluster_config(const std::string& json, const std::string& what) {
  dgpp::minijson::ParseResult parsed;
  try {
    parsed = dgpp::minijson::parse(json);
  } catch (const std::exception& e) {
    fail(what, std::string("invalid JSON: ") + e.what());
  }
  const Value& root = parsed.root;
  if (!root.is_object()) fail(what, "the document must be an object");
  ClusterConfig c;
  int http_override = 0;
  bool saw_model = false, saw_nodes = false;
  for (const Member& m : root.members()) {
    const std::string& k = m.key;
    const Value& v = m.value;
    if (k == "model") {
      c.model = text(v, k, what);
      if (c.model.empty()) fail(what, "'model' must not be empty");
      saw_model = true;
    } else if (k == "nodes") {
      if (!v.is_array() || v.items().empty())
        fail(what, "'nodes' must be a non-empty array of hosts (rank = index)");
      for (const Value& n : v.items()) {
        const std::string host = text(n, "nodes[]", what);
        if (host.empty()) fail(what, "'nodes' must not contain an empty host");
        c.nodes.push_back(host);
      }
      saw_nodes = true;
    } else if (k == "world_size") {
      if (root.find("nodes"))
        fail(what, "'nodes' and 'world_size' are mutually exclusive");
      const int64_t ws = integer(v, k, what, 1, 1024);
      if (ws != 1)
        fail(what, "multi-node configurations require explicit 'nodes'; "
                   "resolve the deployment through scripts/dgpp-cluster");
      c.nodes.push_back("localhost");
      saw_nodes = true;
    } else if (k == "ssh_user") {
      c.ssh_user = text(v, k, what);
    } else if (k == "release") {
      c.release = text(v, k, what);
    } else if (k == "http") {
      if (!v.is_object()) fail(what, "'http' must be an object");
      for (const Member& p : v.members()) {
        if (p.key == "bind_host") {
          c.http_bind = text(p.value, "http.bind_host", what);
          in_addr address{};
          if (::inet_pton(AF_INET, c.http_bind.c_str(), &address) != 1)
            fail(what, "'http.bind_host' must be an IPv4 address");
        } else if (p.key == "port") {
          http_override = static_cast<int>(integer(p.value, "http.port", what, 1, 65535));
        } else if (p.key == "max_body_bytes") {
          // minijson falls back to double for integers outside int64; do not
          // let as_int() convert an out-of-range floating-point value.
          if (p.value.kind() != Value::Kind::Int)
            fail(what, "'http.max_body_bytes' must be an integer (positive 64-bit byte count)");
          c.http_max_body_bytes = integer(p.value, "http.max_body_bytes", what, 1,
                                          std::numeric_limits<int64_t>::max());
        } else fail(what, "unknown key 'http." + p.key + "'");
      }
    } else if (k == "node_env") {
      if (!v.is_array()) fail(what, "'node_env' must be an array");
      for (const Value& node : v.items()) {
        if (!node.is_object()) fail(what, "'node_env[]' must be an object");
        std::map<std::string, std::string> env;
        // The launcher's NODE_KEYS (scripts/site_env.py): the fabric's per-node
        // settings and the L2 weight-prefetch knobs (src/kernels/l2_prefetch.hpp),
        // which an A/B sets the same way on every rank.
        static const char* const kNodeKeys[] = {
            "DGPP_ROCE_DEVICES", "DGPP_ROCE_GID_INDICES", "HF_HUB_CACHE", "DGPP_RESIDENT_CACHE_DIR",
            "DGPP_LOG_LEVEL", "DGPP_MLOCK",
            "DGPP_L2_PREFETCH", "DGPP_L2_PREFETCH_MB", "DGPP_L2_PREFETCH_BOUNDARY", "DGPP_L2_PREFETCH_LAYER",
            // The bus timeline switch and the dense-lowering A/B switches: every rank the same.
            "DGPP_BUS_TIMELINE", "DGPP_DSV41_DENSE_GEMV", "DGPP_DENSE_GEMV_ROWS", "DGPP_DSV41_EAGER_FOLD"};
        for (const Member& setting : node.members()) {
          bool known = false;
          for (const char* key : kNodeKeys) known = known || setting.key == key;
          if (!known) fail(what, "unknown node environment key '" + setting.key + "'");
          env[setting.key] = text(setting.value, "node_env." + setting.key, what);
        }
        c.node_env.push_back(std::move(env));
      }
    } else if (k == "ports") {
      if (!v.is_object()) fail(what, "'ports' must be an object");
      for (const Member& p : v.members()) {
        const std::string pk = "ports." + p.key;
        if (p.key == "http") c.http_port = static_cast<int>(integer(p.value, pk, what, 1, 65535));
        else if (p.key == "fabric") c.fabric_port = static_cast<int>(integer(p.value, pk, what, 1, 65535));
        else if (p.key == "journal") c.journal_port = static_cast<int>(integer(p.value, pk, what, 1, 65535));
        else fail(what, "unknown key '" + pk + "'");
      }
    } else if (k == "engine") {
      if (!v.is_object()) fail(what, "'engine' must be an object");
      ClusterConfig::Engine& e = c.engine;
      for (const Member& p : v.members()) {
        const std::string ek = "engine." + p.key;
        const Value& x = p.value;
        if (p.key == "max_concurrency") e.max_concurrency = static_cast<int>(integer(x, ek, what, 1, 1 << 20));
        else if (p.key == "kv_capacity") e.kv_capacity = integer(x, ek, what, 1, 1ll << 40);
        else if (p.key == "kv_dtype") {
          e.kv_dtype = text(x, ek, what);
          if (!latent_format_from_string(e.kv_dtype))
            fail(what, "'" + ek + "' must be \"bf16\", \"fp8\" or \"fp4\"");
        }
        else if (p.key == "embed_sharding") {
          e.embed_sharding = text(x, ek, what);
          if (e.embed_sharding != "replicated" && e.embed_sharding != "vocab")
            fail(what, "'" + ek + "' must be \"replicated\" or \"vocab\"");
        }
        else if (p.key == "ngram_table") {
          e.ngram_table = text(x, ek, what);
          if (e.ngram_table != "resident" && e.ngram_table != "mmap")
            fail(what, "'" + ek + "' must be \"resident\" or \"mmap\"");
        }
        else if (p.key == "dense_weights") {
          e.dense_weights = text(x, ek, what);
          if (e.dense_weights != "checkpoint" && e.dense_weights != "fp8")
            fail(what, "'" + ek + "' must be \"checkpoint\" or \"fp8\"");
        }
        else if (p.key == "fp8_head") {
          e.fp8_head = text(x, ek, what);
          if (e.fp8_head != "gemv" && e.fp8_head != "mma")
            fail(what, "'" + ek + "' must be \"gemv\" or \"mma\"");
        }
        else if (p.key == "mtp_expert_format") {
          e.mtp_expert_format = text(x, ek, what);
          if (e.mtp_expert_format != "fp8" && e.mtp_expert_format != "bf16_fused")
            fail(what, "'" + ek + "' must be \"fp8\" or \"bf16_fused\"");
        }
        else if (p.key == "bf16_weights") {
          e.bf16_weights = text(x, ek, what);
          if (!parse_bf16_residency(e.bf16_weights, nullptr))
            fail(what, "'" + ek + "' must be \"checkpoint\", \"bf12\" or \"bf12+bf16\"");
        } else if (p.key == "prefill") {
          e.prefill = text(x, ek, what);
          if (e.prefill != "bounded" && e.prefill != "exact")
            fail(what, "'" + ek + "' must be \"bounded\" or \"exact\"");
        } else if (p.key == "rope_scaling") {
          if (!x.is_object()) fail(what, "'" + ek + "' must be an object");
          dgpp::RopeScaling rs;
          bool saw_factor = false, saw_original = false;
          for (const Member& q : x.members()) {
            const std::string rk = ek + "." + q.key;
            const Value& y = q.value;
            if (q.key == "rope_type" || q.key == "type") {
              const std::string rt = text(y, rk, what);
              if (rt != "yarn")
                fail(what, "'" + rk + "' must be \"yarn\" (the only ramp this engine builds)");
            } else if (q.key == "factor") {
              rs.factor = number(y, rk, what);
              saw_factor = true;
            } else if (q.key == "original_max_position_embeddings") {
              rs.original_max_position_embeddings = integer(y, rk, what, 1, 1ll << 40);
              saw_original = true;
            } else if (q.key == "beta_fast") {
              rs.beta_fast = number(y, rk, what);
            } else if (q.key == "beta_slow") {
              rs.beta_slow = number(y, rk, what);
            } else if (q.key == "attn_factor") {
              rs.attn_factor = number(y, rk, what);
            } else if (q.key == "mrope_cache_factor") {
              rs.mrope_cache_factor = number(y, rk, what);
            } else {
              fail(what, "unknown key '" + rk + "'");
            }
          }
          if (!saw_factor) fail(what, "'" + ek + ".factor' is required");
          if (!saw_original)
            fail(what, "'" + ek + ".original_max_position_embeddings' is required");
          try {
            rs.validate(ek);
          } catch (const std::runtime_error& err) {
            fail(what, err.what());
          }
          e.rope_scaling = rs;
        } else if (p.key == "default_max_tokens")
          e.default_max_tokens = static_cast<int>(integer(x, ek, what, 1, 1 << 30));
        else if (p.key == "file_inputs") {
          if (!x.is_object()) fail(what, "engine.file_inputs must be an object");
          for (const auto& f : x.members()) {
            const auto path = ek + "." + f.key;
            if (f.key == "directory" || f.key == "pdf_command") {
              if (!f.value.is_string()) fail(what, path + " must be a string");
              (f.key == "directory" ? e.file_inputs.directory : e.file_inputs.pdf_command) = expand_home(std::string(f.value.as_string()));
            } else if (f.key == "max_file_bytes") e.file_inputs.max_file_bytes = integer(f.value, path, what, 1, INT64_MAX);
            else if (f.key == "max_request_bytes") e.file_inputs.max_request_bytes = integer(f.value, path, what, 1, INT64_MAX);
            else if (f.key == "max_text_bytes") e.file_inputs.max_text_bytes = integer(f.value, path, what, 1, INT64_MAX);
            else if (f.key == "max_storage_bytes") e.file_inputs.max_storage_bytes = integer(f.value, path, what, 1, INT64_MAX);
            else if (f.key == "pdf_timeout_ms") e.file_inputs.pdf_timeout_ms = static_cast<int>(integer(f.value, path, what, 1, INT32_MAX));
            else if (f.key == "workers") e.file_inputs.workers = static_cast<int>(integer(f.value, path, what, 1, 256));
            else fail(what, "unknown key '" + path + "'");
          }
        }
        else if (p.key == "queue_limit") e.queue_limit = static_cast<int>(integer(x, ek, what, 1, 1 << 30));
        else if (p.key == "max_connections") e.max_connections = static_cast<int>(integer(x, ek, what, 1, 1 << 20));
        else if (p.key == "no_eos") e.no_eos = boolean(x, ek, what);
        else if (p.key == "decode_graph") e.decode_graph = boolean(x, ek, what);
        else if (p.key == "mtp") e.mtp = boolean(x, ek, what);
        else if (p.key == "mtp_depth") {
          e.mtp_depth = static_cast<int>(integer(x, ek, what, 1, 5));  // kSpecRows - 1
          e.mtp_depth_set = true;
        } else if (p.key == "compact_batches")
          e.compact_batches = boolean(x, ek, what);
        else if (p.key == "graph_batch_min_live") e.graph_batch_min_live = static_cast<int>(integer(x, ek, what, 0, 1 << 20));
        else if (p.key == "mtp_schedule") e.mtp_schedule = boolean(x, ek, what);
        else if (p.key == "mtp_schedule_row_ms") {
          e.mtp_schedule_row_ms = number(x, ek, what);
          if (!(e.mtp_schedule_row_ms > 0.0)) fail(what, "'" + ek + "' must be > 0");
        } else if (p.key == "mtp_schedule_base_ms") {
          e.mtp_schedule_base_ms = number(x, ek, what);
          if (!(e.mtp_schedule_base_ms >= 0.0)) fail(what, "'" + ek + "' must be >= 0");
        } else if (p.key == "mtp_schedule_lambda") {
          e.mtp_schedule_lambda = number(x, ek, what);
          if (!(e.mtp_schedule_lambda >= 0.0)) fail(what, "'" + ek + "' must be >= 0 (0: the reservation rate)");
        } else if (p.key == "mtp_schedule_min_depth") e.mtp_schedule_min_depth = static_cast<int>(integer(x, ek, what, 1, 5));
        else if (p.key == "mtp_schedule_adapt") e.mtp_schedule_adapt = boolean(x, ek, what);
        else if (p.key == "sampling_candidates") e.sampling_candidates = static_cast<int>(integer(x, ek, what, 1, 256));
        else if (p.key == "prefix_cache_gib") {
          e.prefix_cache_gib = number(x, ek, what);
          if (e.prefix_cache_gib < 0.0) fail(what, "'" + ek + "' must be >= 0 (0 turns the cache off)");
        } else if (p.key == "admission") {
          e.admission = text(x, ek, what);
          if (e.admission != "full" && e.admission != "grow")
            fail(what, "'" + ek + "' must be \"full\" or \"grow\"");
        } else if (p.key == "admission_window") e.admission_window = static_cast<int>(integer(x, ek, what, 1, 1 << 30));
        else if (p.key == "prefill_budget_tokens") e.prefill_budget_tokens = static_cast<int>(integer(x, ek, what, -1, 1 << 30));
        else if (p.key == "model_alias") e.model_alias = text(p.value, ek, what);
        else if (p.key == "prefill_idle_budget_tokens") e.prefill_idle_budget_tokens = static_cast<int>(integer(x, ek, what, 0, 1 << 30));
        else if (p.key == "bulk_pace_gbps") e.bulk_pace_gbps = number(x, ek, what);
        else if (p.key == "bulk_inflight") e.bulk_inflight = static_cast<int>(integer(x, ek, what, -1, 1 << 20));
        else if (p.key == "rendezvous_timeout_ms") e.rendezvous_timeout_ms = static_cast<int>(integer(x, ek, what, 1, 1 << 30));
        else if (p.key == "stats_interval_s") {
          e.stats_interval_s = number(x, ek, what);
          if (e.stats_interval_s < 0.0) fail(what, "'" + ek + "' must be >= 0 (0 turns the line off)");
        } else if (p.key == "reasoning_in_content") e.reasoning_in_content = boolean(x, ek, what);
        else fail(what, "unknown key '" + ek + "'");
      }
    } else if (k == "nvme_cache") {
      if (!v.is_object()) fail(what, "'nvme_cache' must be an object");
      ClusterConfig::NvmeCache& n = c.nvme_cache;
      for (const Member& p : v.members()) {
        const std::string nk = "nvme_cache." + p.key;
        if (p.key == "enabled") n.enabled = boolean(p.value, nk, what);
        else if (p.key == "path") {
          n.path = text(p.value, nk, what);
          if (n.path.empty()) fail(what, "'" + nk + "' must not be empty");
        } else if (p.key == "capacity_gib") {
          n.capacity_gib = number(p.value, nk, what);
          if (n.capacity_gib < 0.0) fail(what, "'" + nk + "' must be >= 0");
        } else if (p.key == "min_tokens") n.min_tokens = integer(p.value, nk, what, 0, 1ll << 40);
        else fail(what, "unknown key '" + nk + "'");
      }
      if (n.enabled && n.capacity_gib <= 0.0)
        fail(what, "'nvme_cache.capacity_gib' must be positive when the cache is enabled");
    } else if (k == "paths") {
      if (!v.is_object()) fail(what, "'paths' must be an object");
      for (const Member& p : v.members()) {
        const std::string pk = "paths." + p.key;
        if (p.key == "log_dir") c.paths.log_dir = text(p.value, pk, what);
        else if (p.key == "stage_dir") c.paths.stage_dir = text(p.value, pk, what);
        else if (p.key == "release_dir") c.paths.release_dir = text(p.value, pk, what);
        else if (p.key == "resident_cache") c.paths.resident_cache = text(p.value, pk, what);
        else fail(what, "unknown key '" + pk + "'");
      }
    } else {
      fail(what, "unknown key '" + k + "'");
    }
  }
  if (!saw_model) fail(what, "'model' is required");
  if (http_override) c.http_port = http_override;
  if (!saw_nodes) fail(what, "'nodes' is required");
  if (!c.node_env.empty() && c.node_env.size() != c.nodes.size())
    fail(what, "'node_env' must have one entry per node");
  if (c.fabric_port == c.journal_port)
    fail(what, "'ports.fabric' and 'ports.journal' must differ");
  return c;
}

ClusterConfig load_cluster_config(const std::string& path) {
  std::ifstream in(expand_home(path));
  if (!in) throw std::runtime_error("cluster config " + path + ": cannot read");
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_cluster_config(ss.str(), path);
}

}  // namespace dgpp::serve
