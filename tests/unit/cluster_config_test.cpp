// The cluster config: the schema is checked by name, the
// engine defaults are the binary's own, the launcher's resolved configuration
// parses, and the digest is stable and sensitive.
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "serve/cluster_config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string refusal(const std::string& json) {
  try {
    (void)dgpp::serve::parse_cluster_config(json, "t");
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

// A minimal deployment with one engine override spliced in.
std::string engine_json(const std::string& engine) {
  return "{\"model\":\"m\",\"nodes\":[\"h\",\"w\"],\"engine\":" + engine + "}";
}

}  // namespace

DGPP_TEST(cluster_config_parses_engine_rope_scaling) {
  // Absent (the default): the Qwen family's plain rope, bit for bit what
  // every earlier build served.
  const dgpp::serve::ClusterConfig off =
      dgpp::serve::parse_cluster_config(engine_json("{\"kv_capacity\":8192}"), "t");
  require(!off.engine.rope_scaling.has_value(), "off by default");
  // The recipe's knob: factor 2 over the release's 262144 positions. The
  // correction band's mrope enlargement (4) and the attention factor (1)
  // are the defaults vLLM runs with, so they are absent here too.
  const dgpp::serve::ClusterConfig on = dgpp::serve::parse_cluster_config(
      engine_json("{\"rope_scaling\":{\"factor\":2.0,"
                  "\"original_max_position_embeddings\":262144}}"),
      "t");
  require(on.engine.rope_scaling.has_value(), "the knob is parsed");
  require(on.engine.rope_scaling->correction_max_position() == 1048576,
          "the band's default mrope cache is 4x");
  require(on.engine.rope_scaling->context_limit() == 524288, "the 512K ceiling");
  require(on.engine.rope_scaling->beta_fast == 32.0 &&
              on.engine.rope_scaling->beta_slow == 1.0 &&
              on.engine.rope_scaling->attn_factor == 1.0,
          "vLLM's YaRN defaults");
  // Every field is settable and checked by name.
  const dgpp::serve::ClusterConfig full = dgpp::serve::parse_cluster_config(
      engine_json("{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,"
                  "\"original_max_position_embeddings\":262144,\"beta_fast\":16,"
                  "\"beta_slow\":2,\"attn_factor\":1.5,\"mrope_cache_factor\":1}}"),
      "t");
  require(full.engine.rope_scaling->factor == 4.0 && full.engine.rope_scaling->beta_fast == 16.0 &&
              full.engine.rope_scaling->beta_slow == 2.0 &&
              full.engine.rope_scaling->attn_factor == 1.5 &&
              full.engine.rope_scaling->mrope_cache_factor == 1.0 &&
              full.engine.rope_scaling->context_limit() == 1048576,
          "every field");
  require(refusal(engine_json("{\"rope_scaling\":{\"factor\":0.5,"
                              "\"original_max_position_embeddings\":262144}}"))
                  .find("factor") != std::string::npos,
          "a sub-unity factor refused by name");
  require(refusal(engine_json("{\"rope_scaling\":{\"factor\":2.0}}")).find(
              "original_max_position_embeddings") != std::string::npos,
          "a missing original context refused by name");
  require(refusal(engine_json("{\"rope_scaling\":{\"original_max_position_embeddings\":262144}}"))
                  .find("factor") != std::string::npos,
          "a missing factor refused by name");
  require(refusal(engine_json("{\"rope_scaling\":{\"factor\":2.0,"
                              "\"original_max_position_embeddings\":262144,"
                              "\"mscale\":2.0}}"))
                  .find("mscale") != std::string::npos,
          "an unknown field refused by name (vLLM spells it attn_factor)");
  require(refusal(engine_json("{\"rope_scaling\":[2.0]}")).find("object") != std::string::npos,
          "a non-object refused");
  require(refusal(engine_json("{\"rope_scaling\":{\"rope_type\":\"linear\",\"factor\":2.0,"
                              "\"original_max_position_embeddings\":262144}}"))
                  .find("rope_type") != std::string::npos,
          "only yarn");
}

DGPP_TEST(cluster_config_parses_fills_defaults_and_derives_the_world) {
  const std::string json = R"({
    "model": "org/name",
    "nodes": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    "ssh_user": "ops",
    "release": "0.1.0+gabc",
    "ports": {"http": 8081, "journal": 29001},
    "engine": {"max_concurrency": 2, "decode_graph": true, "prefix_cache_gib": 0.5,
               "admission": "grow", "stats_interval_s": 0, "mtp_depth": 2, "prefill": "exact",
               "prefill_budget_tokens": 256, "prefill_idle_budget_tokens": 2048},
    "paths": {"log_dir": "/var/log/dgpp"}
  })";
  const dgpp::serve::ClusterConfig c = dgpp::serve::parse_cluster_config(json, "t");
  require(c.model == "org/name" && c.world() == 3 && c.nodes[0] == "10.0.0.1" &&
              c.nodes[2] == "10.0.0.3" && c.ssh_user == "ops" && c.release == "0.1.0+gabc",
          "the model, the nodes, the user and the release");
  require(c.http_port == 8081 && c.fabric_port == 29970 && c.journal_port == 29001,
          "the ports: given ones taken, the fabric port defaulted");
  require(c.engine.max_concurrency == 2 && c.engine.decode_graph && !c.engine.mtp &&
              c.engine.mtp_depth == 2 && c.engine.prefix_cache_gib == 0.5 &&
              c.engine.admission == "grow" && c.engine.stats_interval_s == 0.0 && c.engine.prefill == "exact" &&
              c.engine.prefill_budget_tokens == 256 && c.engine.prefill_idle_budget_tokens == 2048,
          "the given engine knobs");
  // The engine defaults are the binary's flag defaults — one set of defaults.
  require(c.engine.kv_capacity == 8192 && c.engine.default_max_tokens == 256 &&
              c.engine.queue_limit == 64 && c.engine.max_connections == 64 &&
              !c.engine.no_eos && c.engine.graph_batch_min_live == 0 &&
              c.engine.sampling_candidates == 128 && c.engine.admission_window == 256 &&
              c.engine.bulk_pace_gbps == -1.0 && c.engine.bulk_inflight == -1 &&
              c.engine.rendezvous_timeout_ms == 120000 && !c.engine.reasoning_in_content &&
              c.engine.kv_dtype == "bf16",
          "the engine defaults");
  // The KV dtype: named by the config, checked by name.
  const dgpp::serve::ClusterConfig fp8 = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"kv_dtype":"fp8"}})", "t");
  require(fp8.engine.kv_dtype == "fp8", "kv_dtype fp8");
  require(c.paths.log_dir == "/var/log/dgpp" && c.paths.stage_dir == "/tmp/bus4" &&
              c.paths.release_dir == "~/dgpp/releases" && c.paths.resident_cache.empty(),
          "the paths: given one taken, the rest defaulted");
  // A one-node config is a world of one.
  const dgpp::serve::ClusterConfig one =
      dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t");
  require(one.world() == 1 && one.http_port == 18080, "a one-node world");
  require(one.http_bind == "127.0.0.1", "HTTP defaults to loopback");
  require(one.http_max_body_bytes == 256ll * 1024 * 1024, "HTTP body default supports large prefills");
}

DGPP_TEST(cluster_config_http_override_is_order_independent) {
  const auto c = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"http":{"bind_host":"0.0.0.0","port":8080,"max_body_bytes":5368709120},"ports":{"http":18080},"node_env":[{"HF_HUB_CACHE":"~/cache"}]})", "t");
  require(c.http_bind == "0.0.0.0" && c.http_port == 8080, "deployment HTTP wins over legacy ports");
  require(c.http_max_body_bytes == 5368709120ll, "body limit retains a 64-bit byte count");
  require(c.node_env.at(0).at("HF_HUB_CACHE") == "~/cache", "node cache retained");
  require(!refusal(R"({"model":"m","nodes":["h"],"http":{"bind_host":"localhost"}})").empty(), "bind must be IPv4");
  require(!refusal(R"({"model":"m","nodes":["h"],"node_env":[{"HF_TOKEN":"x"}]})").empty(), "credentials disallowed");
}

DGPP_TEST(cluster_config_fileInputLimitsAndPaths) {
  const auto c = dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"],"engine":{
    "file_inputs":{"directory":"~/dgpp/input-files","pdf_command":"/usr/bin/pdftotext",
      "max_file_bytes":5368709120,"max_request_bytes":6442450944,"max_text_bytes":1073741824,
      "max_storage_bytes":1099511627776,"pdf_timeout_ms":300000,"workers":8}}})", "t");
  const auto& f = c.engine.file_inputs;
  require(f.directory == dgpp::serve::expand_home("~/dgpp/input-files") && f.pdf_command == "/usr/bin/pdftotext",
          "file paths reach the deployment with home expanded");
  require(f.max_file_bytes == 5368709120ull && f.max_request_bytes == 6442450944ull &&
          f.max_text_bytes == 1073741824ull && f.max_storage_bytes == 1099511627776ull &&
          f.pdf_timeout_ms == 300000 && f.workers == 8, "large file limits retain 64-bit byte counts");
  for (const auto& value : {R"({"workers":0})", R"({"workers":257})", R"({"max_file_bytes":-1})",
                           R"({"max_text_bytes":1.5})", R"({"pdf_timeout_ms":0})", R"({"unknown":1})"}) {
    require(refusal(std::string(R"({"model":"m","nodes":["h"],"engine":{"file_inputs":)") + value + "}}")
                .find("engine.file_inputs.") != std::string::npos, "invalid file settings report their path");
  }
}

DGPP_TEST(cluster_config_refusesUnknownKeysAndBadValuesByName) {
  const struct {
    const char* json;
    const char* needle;
  } cases[] = {
      {R"({"nodes":["h"]})", "'model' is required"},
      {R"({"model":"m"})", "'nodes' is required"},
      {R"({"model":"m","nodes":[]})", "'nodes' must be a non-empty array"},
      {R"({"model":"m","nodes":["h"],"nodez":1})", "unknown key 'nodez'"},
      {R"({"model":"m","nodes":["h"],"engine":{"max_concurency":2}})", "unknown key 'engine.max_concurency'"},
      {R"({"model":"m","nodes":["h"],"engine":{"max_concurrency":"2"}})", "'engine.max_concurrency' must be an integer"},
      {R"({"model":"m","nodes":["h"],"engine":{"max_concurrency":1.5}})", "'engine.max_concurrency' must be an integer"},
      {R"({"model":"m","nodes":["h"],"engine":{"max_concurrency":0}})", "'engine.max_concurrency' must be in [1,"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp":"yes"}})", "'engine.mtp' must be true or false"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_depth":6}})", "'engine.mtp_depth' must be in [1, 5]"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_depth":0}})", "'engine.mtp_depth' must be in [1, 5]"},
      {R"({"model":"m","nodes":["h"],"engine":{"admission":"fast"}})", "'engine.admission' must be \"full\" or \"grow\""},
      {R"({"model":"m","nodes":["h"],"engine":{"prefix_cache_gib":-1}})", "'engine.prefix_cache_gib' must be >= 0"},
      {R"({"model":"m","nodes":["h"],"engine":{"kv_dtype":"int8"}})", "'engine.kv_dtype' must be \"bf16\", \"fp8\" or \"fp4\""},
      {R"({"model":"m","nodes":["h"],"engine":{"kv_dtype":8}})", "'engine.kv_dtype' must be a string"},
      {R"({"model":"m","nodes":["h"],"engine":{"prefill":"fast"}})", "'engine.prefill' must be \"bounded\" or \"exact\""},
      {R"({"model":"m","nodes":["h"],"ports":{"http":70000}})", "'ports.http' must be in [1, 65535]"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":0}})", "'http.max_body_bytes' must be in [1,"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":-1}})", "'http.max_body_bytes' must be in [1,"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":true}})", "'http.max_body_bytes' must be an integer"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":"256MiB"}})", "'http.max_body_bytes' must be an integer"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":1.5}})", "'http.max_body_bytes' must be an integer"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":9223372036854775808}})", "http.max_body_bytes"},
      {R"({"model":"m","nodes":["h"],"ports":{"fabric":5,"journal":5}})", "'ports.fabric' and 'ports.journal' must differ"},
      {R"({"model":"m","nodes":["h"],"paths":{"logs":"/x"}})", "unknown key 'paths.logs'"},
      {R"({"model":"m","nodes":["h"],"paths":{"log_dir":3}})", "'paths.log_dir' must be a string"},
      {R"({"model":"m","nodes":["h"],)", "invalid JSON"},
      {R"([1,2])", "must be an object"},
  };
  for (const auto& c : cases) {
    const std::string what = refusal(c.json);
    require(what.find(c.needle) != std::string::npos,
            std::string("expected a refusal naming ") + c.needle + " for " + c.json +
                " — got: " + what);
  }
}

DGPP_TEST(cluster_config_theResolvedFileParsesAndTheDigestIsStable) {
  // site_env_test.py checks this fixture against the deployment resolver.
  const dgpp::serve::ClusterConfig ex = dgpp::serve::load_cluster_config(
      std::string(DGPP_SOURCE_DIR) + "/tests/fixtures/cluster.resolved.json");
  require(ex.world() == 4 && ex.model == "HawkBearPig/GLM-5.3-Flash-NVFP4-FP8" &&
              ex.http_port == 18080 && ex.engine.max_concurrency == 4 &&
              ex.engine.kv_capacity == 786432 && ex.engine.queue_limit == 8 &&
              ex.engine.kv_dtype == "bf16" &&
              ex.engine.decode_graph && ex.engine.mtp && ex.release.empty() &&
              ex.ssh_user == "ops",
          "the resolved deployment has the model settings and example site values");
  require(dgpp::serve::config_digest("a") == dgpp::serve::config_digest("a") &&
              dgpp::serve::config_digest("a") != dgpp::serve::config_digest("b") &&
              dgpp::serve::config_digest("x").size() == 16,
          "the digest is stable, sensitive and 16 hex digits");
  require(dgpp::serve::expand_home("~/x") != "~/x" &&
              dgpp::serve::expand_home("/abs") == "/abs" &&
              dgpp::serve::expand_home("~user/x") == "~user/x",
          "~ expands, ~user and absolute paths do not change");
  try {
    (void)dgpp::serve::load_cluster_config("/nonexistent/cluster.json");
    require(false, "a missing file must throw");
  } catch (const std::runtime_error& e) {
    require(std::string(e.what()).find("cannot read") != std::string::npos, e.what());
  }
}

DGPP_TEST(cluster_config_the_yarn512k_template_is_a_deployment_the_engine_reads) {
  // deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json, the
  // ready-to-run 512K shape (review item 6, 2026-09-18). A template is the
  // launcher's input, so the one edit applied here is the resolver's own:
  // `world_size` becomes that many `nodes` (site_env_test.py checks that leg,
  // portability_test.py the filename). What this checks is the leg that matters
  // for the knob — every field name in the file is one this parser reads (an
  // unknown one is a hard failure), the ramp it builds reaches the 512K
  // ceiling, and the pool the template seats requests in clears it. A YaRN
  // ramp raises the positional ceiling and enlarges nothing else, so a ceiling
  // above the pool would be a template whose longest request its own pool
  // refuses: the invariant the last two lines check.
  const std::string path = std::string(DGPP_SOURCE_DIR) +
                           "/deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json";
  std::ifstream in(path);
  require(in.good(), "the tracked template is readable: " + path);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::smatch world;
  require(std::regex_search(text, world, std::regex("\"world_size\"\\s*:\\s*([0-9]+)")),
          "the template names its world");
  std::string nodes = "\"nodes\": [";
  for (int i = 0; i < std::stoi(world[1]); ++i)
    nodes += (i ? ", " : "") + std::string("\"rank") + std::to_string(i) + "\"";
  nodes += "]";
  const dgpp::serve::ClusterConfig c = dgpp::serve::parse_cluster_config(
      std::regex_replace(text, std::regex("\"world_size\"\\s*:\\s*[0-9]+"), nodes), path);
  const dgpp::RopeScaling& rs = *c.engine.rope_scaling;
  require(c.model == "nvidia/Qwen3.8-Flash-Next-NVFP4" && c.world() == 2,
          "the two-Spark Qwen deployment");
  require(c.engine.rope_scaling.has_value() && rs.factor == 2.0 &&
              rs.original_max_position_embeddings == 262144 && rs.beta_fast == 32.0 &&
              rs.beta_slow == 1.0 && rs.attn_factor == 1.0 && rs.mrope_cache_factor == 4.0 &&
              rs.context_limit() == 524288 && rs.correction_max_position() == 1048576,
          "the recipe's ramp, spelled out field by field");
  require(c.engine.decode_graph && c.engine.mtp && c.engine.max_concurrency == 2,
          "the template rules (MTP, the decode graph) and two request slots");
  require(c.engine.kv_capacity % 64 == 0, "a whole number of KV blocks");
  require(c.engine.kv_capacity > rs.context_limit() + c.engine.default_max_tokens,
          "the pool clears the ceiling with room for an answer: a 524288-token "
          "request must be admissible in this template");
}
