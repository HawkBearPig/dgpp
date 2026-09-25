// Resolved runtime configuration shared by the launcher and server.
//
// The launcher combines a deployment JSON (model, world_size, engine options)
// with site settings from .env. This loader reads the resulting JSON, with
// explicit nodes, ports and paths. nodes[0] is rank 0, which serves HTTP and sends
// effective model/engine settings to peers through the admission journal.
// Peers read bootstrap addresses and local paths from their own files.
//
// Flags after --config override the file. Unknown keys and invalid types
// are errors. Engine defaults match the binary's defaults; deployment
// templates set serving values explicitly. Each rank hashes its effective
// shared configuration, and the warm record checks those hashes before
// serving. See README.md for the schema and deploy/*.example.json for
// templates. Multi-node deployment templates must be resolved before this loader
// reads them. A single-node template may use world_size: 1 directly (localhost);
// nodes and world_size are mutually exclusive.
#pragma once
#include "serve/file_inputs.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "kernels/rope_scaling.hpp"
#include "serve/http_limits.hpp"

namespace dgpp::serve {

struct ClusterConfig {
  std::string model;
  std::vector<std::string> nodes;  // rank = index; nodes[0] is the head
  std::string ssh_user;            // empty: the launcher's own user
  std::string release;             // the installed release the launcher runs (launcher-only; empty: the development binary)
  int http_port = 18080;
  std::string http_bind = "127.0.0.1";
  int64_t http_max_body_bytes = kDefaultHttpMaxBodyBytes;
  // Local paths and device names may differ by rank; no credentials belong here.
  std::vector<std::map<std::string, std::string>> node_env;
  int fabric_port = 29970;
  int journal_port = 29971;
  struct Engine {
    int max_concurrency = 8;
    int64_t kv_capacity = 8192;
    std::string kv_dtype = "bf16";  // the latent cache's format: bf16 | fp8 | fp4
    // The full GLM-5.3's embedding table: "replicated" on every rank, or
    // "vocab" — each rank holds its lm-head slice of the rows, gathered
    // per token and summed by one fold (bitwise the same numbers, one
    // small collective per lookup, −1.33 GiB per rank at world 4).
    std::string embed_sharding = "replicated";
    // The Qwen n-gram table's residency: "resident" copies it
    // to the device (the default; 47.7 GiB at world 1), "mmap" leaves it
    // on the NVMe behind the page cache and gathers each step's rows on
    // the host — the single-Spark deployment.
    std::string ngram_table = "resident";
    // The Qwen dense stack's form: "checkpoint" (the default:
    // the BF16 the checkpoint ships) or "fp8" (every dense projection
    // encoded to block FP8 at load — the same recipe as the FP8 releases;
    // docs/qwen38_single_spark.md).
    std::string dense_weights = "checkpoint";
    std::string fp8_head = "gemv";  // Qwen head: gemv | mma (opt-in)
    // The MTP draft layer's routed-expert layout: "fp8" (the default:
    // per-expert FP8 tensors as the NVIDIA release ships) or "bf16_fused"
    // (the RadixArk release: all 512 experts stacked as fused BF16
    // gate_up_proj and down_proj, encoded to FP8 at load). Applies to the
    // Qwen family only.
    std::string mtp_expert_format = "fp8";
    // The resident form of the bf16 weights the decode GEMV reads (the
    // attention / linear-attention projections, the lm head):
    // "checkpoint" (the default: the bf16 bytes alone), "bf12" — a lossless
    // 12-bit form of each matrix ALONE (kernels/bf12_gemv.hpp: the
    // sign+mantissa byte and a 4-bit exponent code, bitwise the bf16 GEMV,
    // 0.75 of the bytes a decode step streams and 0.75 of the memory; a
    // prefill GEMM expands the rows it reads into a small scratch) — or
    // "bf12+bf16": both forms resident (the same decode, prefill reads the
    // bf16 bytes in place, +0.75 of those matrices' memory). The memory
    // plan carries either (common/bf16_residency.hpp).
    std::string bf16_weights = "checkpoint";
    // The DeepSeek-V4.1 prefill mode (docs/deepseek_v41_flash_plan.md
    // §1.8): "bounded" (the default: the encoder over every prompt row,
    // the decoder over the last window rows — the model's own serving
    // recipe, half the prefill work) or "exact" (every layer over every
    // row, the parity mode).
    std::string prefill = "bounded";
    // The opt-in YaRN rope ramp (2026-09-18, the Qwen3.8-Flash-Next 512K
    // recipe): absent is the default and leaves every table and every
    // kernel argument exactly as they were; present builds the Qwen QSA
    // rope from the YaRN inverse frequencies, scales its cos/sin by the
    // vLLM attention factor (yarn_get_mscale(factor) * attn_factor) and
    // lifts the context cap from max_position_embeddings to
    // original_max_position_embeddings x factor. The checkpoint's own
    // config never carries it — the NVFP4 release declares no scaling and
    // the A/B target applies YaRN from the launcher's --hf-overrides.
    std::optional<dgpp::RopeScaling> rope_scaling;
    int default_max_tokens = 256;
    FileInputConfig file_inputs;
    int queue_limit = 64;
    int max_connections = 64;
    bool no_eos = false;
    bool compact_batches = false;
    bool decode_graph = false;
    bool mtp = false;
    int mtp_depth = 1;             // draft tokens per step (1..5); needs mtp
    bool mtp_depth_set = false;    // the file named it (else a family may default it: DSpark's block is 5)
    // The confidence-scheduled verify depth (engine/verify_schedule.hpp,
    // 2026-09-14; needs mtp and a family with a confidence head — DSpark):
    // a step verifies only the leading drafts whose prefix survival beats
    // the value of a verify row. The cost curve and the value of time are
    // the world's (every rank takes rank 0's): `row_ms` one verify row,
    // `base_ms` the step's fixed cost (the draft included), `lambda` the
    // value of decode time in tokens/ms (0: the reservation rate
    // 1 / (base + row)); `min_depth` the fewest drafts a step verifies.
    bool mtp_schedule = false;
    double mtp_schedule_row_ms = 8.0;
    double mtp_schedule_base_ms = 28.0;
    double mtp_schedule_lambda = 0.0;
    int mtp_schedule_min_depth = 1;
    bool mtp_schedule_adapt = true;  // lambda follows the modeled throughput, floored at mtp_schedule_lambda
    int graph_batch_min_live = 0;  // 0 = min(2, max_concurrency) (the batch family, 2026-09-07)
    int sampling_candidates = 128;
    double prefix_cache_gib = 1.5;
    std::string admission = "full";
    int admission_window = 256;
    int prefill_budget_tokens = -1;  // automatic on engines with resumable prefill
    std::string model_alias = "";  // override the served model name (display only)
    int prefill_idle_budget_tokens = 0;
    double bulk_pace_gbps = -1.0;  // derived from the port rate
    int bulk_inflight = -1;
    int rendezvous_timeout_ms = 120000;
    double stats_interval_s = 10.0;
    bool reasoning_in_content = false;
  } engine;
  // The NVMe cold tier for the prefix cache (issue #26): a preallocated
  // slab per rank under `path`, sized to `capacity_gib`, that evicted
  // entries spill to and later requests restore from. Top-level, so a
  // deployment turns the tier on with one value; `enabled` requires a
  // positive capacity. Startup checks the path, the free space and the
  // minimum size before anything is allocated. `min_tokens` (0: one
  // prefill chunk) is the shortest entry worth keeping cold.
  struct NvmeCache {
    bool enabled = false;
    std::string path = "~/dgpp/nvme-cache";
    double capacity_gib = 0.0;
    int64_t min_tokens = 0;
  } nvme_cache;
  struct Paths {
    std::string log_dir = "~/dgpp/log";
    std::string stage_dir = "/tmp/bus4";     // where the launcher puts the peers' binary and config
    std::string release_dir = "~/dgpp/releases";
    std::string resident_cache;              // empty: the binary's default (~/.cache/dgpp/resident)
  } paths;

  int world() const { return static_cast<int>(nodes.size()); }
};

// Parses the JSON text; `what` names it in errors. Throws
// std::runtime_error naming the offending key.
ClusterConfig parse_cluster_config(const std::string& json, const std::string& what);
// Reads and parses the file.
ClusterConfig load_cluster_config(const std::string& path);

// "~" and "~/..." to $HOME; anything else unchanged.
std::string expand_home(const std::string& path);

// The 64-bit FNV-1a of a canonical configuration string, as 16 hex digits.
std::string config_digest(const std::string& canonical);

}  // namespace dgpp::serve
