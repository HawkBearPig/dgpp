// The cluster config: the schema is checked by name, the
// engine defaults are the binary's own, the launcher's resolved configuration
// parses, and the digest is stable and sensitive.
#include "serve/cluster_config.hpp"

#include <cmath>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

#include "common/test.hpp"

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

DGPP_TEST(cluster_config_rejects_malformed_number_syntax) {
  for (const char* number : {"1e+", "01"})
    require(refusal(engine_json("{\"repetition_penalty\":" + std::string(number) + "}"))
                    .find("malformed number") != std::string::npos,
            "invalid numeric syntax is refused before field validation");
  const auto config =
      dgpp::serve::parse_cluster_config(engine_json("{\"repetition_penalty\":1e+0}"), "t");
  require(config.engine.repetition_penalty == 1.0f, "valid exponent remains accepted");
}

DGPP_TEST(cluster_config_penalty_defaults_preserve_omission_and_explicit_neutral_values) {
  const auto absent = dgpp::serve::parse_cluster_config(engine_json("{}"), "t").engine;
  require(!absent.repetition_penalty && !absent.frequency_penalty && !absent.presence_penalty,
          "omitted penalty defaults inherit the checkpoint and neutral values");
  const auto active =
      dgpp::serve::parse_cluster_config(
          engine_json(
              R"({"repetition_penalty":1.25,"frequency_penalty":-0.5,"presence_penalty":0.25})"),
          "t")
          .engine;
  require(active.repetition_penalty == 1.25f && active.frequency_penalty == -0.5f &&
              active.presence_penalty == 0.25f,
          "all deployment penalty defaults are parsed");
  const auto neutral =
      dgpp::serve::parse_cluster_config(
          engine_json(R"({"repetition_penalty":1,"frequency_penalty":0,"presence_penalty":-0.0})"),
          "t")
          .engine;
  require(neutral.repetition_penalty == 1.0f && neutral.frequency_penalty == 0.0f &&
              neutral.presence_penalty == 0.0f && std::signbit(*neutral.presence_penalty),
          "explicit neutral values are present and retain signed zero");
  for (const std::string key : {"repetition_penalty", "frequency_penalty", "presence_penalty"}) {
    for (const std::string bad : {"null", "true", "\"0.5\"", "[]", "1e300", "-3"})
      require(refusal(engine_json("{\"" + key + "\":" + bad + "}")).find("engine." + key) !=
                  std::string::npos,
              "invalid penalty default rejected by name: " + key + "=" + bad);
  }
  for (const std::string bad : {"0", "-0.0", "1e-100", "100.1"})
    require(
        refusal(engine_json("{\"repetition_penalty\":" + bad + "}")).find("repetition_penalty") !=
            std::string::npos,
        "nonpositive, underflowing or excessive repetition penalty rejected");
  for (const std::string key : {"frequency_penalty", "presence_penalty"})
    require(refusal(engine_json("{\"" + key + "\":2.01}")).find(key) != std::string::npos,
            "additive penalties are bounded above");
}

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

DGPP_TEST(cluster_config_early_prefill_admission_preserves_automatic_and_explicit_values) {
  require(!dgpp::serve::parse_cluster_config(engine_json("{}"), "t").engine.prefill_early_admission.has_value(),
          "omitted admission policy is resolved from the execution backend");
  require(dgpp::serve::parse_cluster_config(engine_json(R"({"prefill_early_admission":false})"), "t")
               .engine.prefill_early_admission == std::optional<bool>(false),
          "early prefill admission can be explicitly disabled");
  for (const std::string value : {"1", "\"true\"", "null"}) {
    bool rejected = false;
    try {
      (void)dgpp::serve::parse_cluster_config(engine_json("{\"prefill_early_admission\":" + value + "}"), "t");
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "early prefill admission requires a JSON boolean");
  }
}

DGPP_TEST(cluster_config_parses_fills_defaults_and_derives_the_world) {
  const std::string json = R"({
    "model": "org/name",
    "nodes": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    "ssh_user": "ops",
    "release": "0.1.0+gabc",
    "ports": {"http": 8081, "journal": 29001, "metrics": 29002},
    "engine": {"max_concurrency": 2, "decode_graph": true, "prefix_cache_gib": 0.5,
               "admission": "grow", "stats_interval_s": 0, "mtp_depth": 2, "prefill": "exact",
               "prefill_budget_tokens": 256, "prefill_idle_budget_tokens": 2048,
               "prefill_early_admission": true,
               "prefix_min_tokens": 512, "prefix_head_snapshots": false, "mtp_draft": "greedy",
               "mtp_schedule_sampled_scale": 0.5, "mtp_draft_temperature": 0.7, "mtp_verify": "block",
               "prefill_dense": "streaming", "prefill_fold_overlap": false, "decode_pdl": false, "prefill_window": "scalar", "decode_window": "scalar", "collective_transport": "send",
               "dflash_batch_rows": 32, "dflash_weights": "fp8", "prefill_group": false,
               "l2_prefetch": false, "l2_prefetch_form": "touch", "l2_prefetch_full_form": "48x4", "l2_prefetch_window_mib": 8,
               "l2_prefetch_boundary_window_mib": 0, "l2_prefetch_boundary_rate": "full",
               "l2_prefetch_layer_rate": "off", "l2_prefetch_merge": false},
    "paths": {"log_dir": "/var/log/dgpp"}
  })";
  const dgpp::serve::ClusterConfig c = dgpp::serve::parse_cluster_config(json, "t");
  require(c.model == "org/name" && c.world() == 3 && c.nodes[0] == "10.0.0.1" &&
              c.nodes[2] == "10.0.0.3" && c.ssh_user == "ops" && c.release == "0.1.0+gabc",
          "the model, the nodes, the user and the release");
  require(c.http_port == 8081 && c.fabric_port == 29970 && c.journal_port == 29001 && c.metrics_port == 29002,
          "the ports: given ones taken, the fabric port defaulted");
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").metrics_port == 0,
          "the peers' metrics listener is off by default");
  require(c.engine.max_concurrency == 2 && c.engine.decode_graph && !c.engine.mtp &&
              c.engine.mtp_depth == 2 && c.engine.prefix_cache_gib == 0.5 &&
              c.engine.admission == "grow" && c.engine.stats_interval_s == 0.0 && c.engine.prefill == "exact" &&
              c.engine.prefill_budget_tokens == 256 && c.engine.prefill_idle_budget_tokens == 2048 &&
              c.engine.prefill_early_admission == std::optional<bool>(true) &&
              c.engine.prefix_min_tokens == 512 && !c.engine.prefix_head_snapshots &&
              c.engine.mtp_draft == "greedy" && c.engine.mtp_schedule_sampled_scale == 0.5 &&
              c.engine.prefill_dense == "streaming" && !c.engine.prefill_fold_overlap && !c.engine.decode_pdl && c.engine.prefill_window == "scalar" &&
              c.engine.decode_window == "scalar" &&
              c.engine.collective_transport == "send" &&
              c.engine.mtp_draft_temperature == 0.7 && c.engine.mtp_verify == "block" &&
              c.engine.dflash_batch_rows == 32 && c.engine.dflash_weights == "fp8" && !c.engine.prefill_group &&
              !c.engine.l2_prefetch && c.engine.l2_prefetch_form == "touch" && c.engine.l2_prefetch_full_form == "48x4" &&
              c.engine.l2_prefetch_window_mib == 8 &&
              c.engine.l2_prefetch_boundary_window_mib == 0 && c.engine.l2_prefetch_boundary_rate == "full" &&
              c.engine.l2_prefetch_layer_rate == "off" && !c.engine.l2_prefetch_merge,
          "the given engine knobs");
  {
    const dgpp::serve::ClusterConfig d = dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t");
    require(d.engine.l2_prefetch && d.engine.l2_prefetch_form == "load" && d.engine.l2_prefetch_full_form == "16x8" &&
            d.engine.l2_prefetch_window_mib == 12 &&
                d.engine.l2_prefetch_boundary_window_mib == 20 && d.engine.l2_prefetch_boundary_rate == "light" &&
                d.engine.l2_prefetch_layer_rate == "light" && d.engine.l2_prefetch_merge,
            "the L2 prefetcher's defaults");
  }
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.prefill_group,
          "cold prompts group by default");
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.dflash_weights == "checkpoint",
          "the drafter serves its checkpoint's weights by default");
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.dflash_batch_rows == 0,
          "the drafter's batches verify whole blocks by default");
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.mtp_verify == "token",
          "the sampled chain's rule defaults to the token test");
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.mtp_draft_temperature == 1.0,
          "the drawn drafts' temperature defaults to the request's");
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.mtp_schedule_sampled_scale == 0.93,
          "the sampled requests' schedule scale defaults to the measured 0.93");
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.mtp_draft == "auto",
          "the sampled requests' draft rule defaults to the family's");
  require(!dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.lookup_draft,
          "context-lookup drafting is off by default");
  {
    const dgpp::serve::ClusterConfig l = dgpp::serve::parse_cluster_config(
        R"({"model":"m","nodes":["h"],"engine":{"lookup_draft":true,"lookup_nmin":3,"lookup_nstrong":5,"lookup_agree":1,"lookup_tail":2}})",
        "t");
    require(l.engine.lookup_draft && l.engine.lookup_nmin == 3 && l.engine.lookup_nstrong == 5 &&
                l.engine.lookup_agree == 1 && l.engine.lookup_tail == 2,
            "the lookup thresholds parse");
    require(!dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t").engine.lookup_tail,
            "the lookup tail defaults to the MTP block alone");
  }
  // The idle engine's arrival gather: 3 ms by default, 0 disables, bounded.
  require(c.engine.admission_gather_ms == 3, "the arrival gather defaults to 3 ms");
  require(dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"],"engine":{"admission_gather_ms":0}})", "t")
                  .engine.admission_gather_ms == 0,
          "the arrival gather can be turned off");
  {
    bool rejected = false;
    try {
      (void)dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"],"engine":{"admission_gather_ms":5000}})", "t");
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "an arrival gather past one second is refused");
  }
  // The engine defaults are the binary's flag defaults — one set of defaults.
  require(c.engine.kv_capacity == 8192 && c.engine.default_max_tokens == 256 &&
              c.engine.queue_limit == 64 && c.engine.max_connections == 64 &&
              c.engine.compact_batches && !c.engine.no_eos && c.engine.graph_batch_min_live == 0 &&
              c.engine.sampling_candidates == 128 && c.engine.admission_window == 256 &&
              c.engine.bulk_pace_gbps == -1.0 && c.engine.bulk_inflight == -1 &&
              c.engine.rendezvous_timeout_ms == 120000 && !c.engine.reasoning_in_content &&
              c.engine.kv_dtype == "bf16" && c.engine.bf16_weights == "checkpoint" &&
              c.engine.fp8_head == "gemv" && c.engine.draft_head == "exact" && c.engine.draft_head_blocks == 32 &&
              !c.engine.prefill_bf16_partials && !c.engine.prefill_fold_scales &&
              !c.engine.prefill_fp8_gemm && !c.engine.prefill_fp8_per_tensor && c.engine.prefill_activations == "bf16" &&
              c.engine.prefill_attention == "bf16" && c.engine.decode_attention == "bf16" &&
              c.engine.dflash_model.empty() &&
              c.engine.dflash_verify_graph && c.engine.dflash_draft_batch && c.engine.dflash_depth == 0 &&
              c.engine.expert_gemm == "wide" && c.engine.expert_gemm_prefetch == 3 &&
              c.engine.expert_tile_list && !c.engine.expert_gemm_pair && c.engine.ngram_prestage,
          "the engine defaults");
  // The expert GEMM's form and companions (2026-09-30): keys, not environment switches.
  const auto xg = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"expert_gemm":"wide4r","expert_gemm_prefetch":0,"expert_tile_list":false,"expert_gemm_pair":true,"ngram_prestage":false}})",
      "t");
  require(xg.engine.expert_gemm == "wide4r" && xg.engine.expert_gemm_prefetch == 0 && !xg.engine.expert_tile_list &&
              xg.engine.expert_gemm_pair && !xg.engine.ngram_prestage,
          "the expert GEMM keys parse");
  // The opt-in prefill levers (2026-09-30): off unless the config says so.
  for (const std::string key : {"prefill_bf16_partials", "prefill_fold_scales", "prefill_fp8_gemm",
                                "prefill_fp8_per_tensor"}) {
    const auto on = dgpp::serve::parse_cluster_config(
        R"({"model":"m","nodes":["h"],"engine":{")" + key + R"(":true}})", "t");
    const bool got = key == "prefill_bf16_partials" ? on.engine.prefill_bf16_partials
                     : key == "prefill_fold_scales"  ? on.engine.prefill_fold_scales
                     : key == "prefill_fp8_gemm"     ? on.engine.prefill_fp8_gemm
                                                     : on.engine.prefill_fp8_per_tensor;
    const int others = (on.engine.prefill_bf16_partials ? 1 : 0) + (on.engine.prefill_fold_scales ? 1 : 0) +
                       (on.engine.prefill_fp8_gemm ? 1 : 0) + (on.engine.prefill_fp8_per_tensor ? 1 : 0);
    require(got && others == 1, "engine." + key + " opt-in alone");
  }
  // The DeepSeek-V4.1 index budget (2026-10-18): 0 (the checkpoint's) unless the config says.
  require(c.engine.index_topk == 0 && c.engine.candidate_topk_blocks == 0 && c.engine.candidate_block_size == 0,
          "the index budget defaults to the checkpoint's");
  const auto ib = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"index_topk":2048,"candidate_topk_blocks":1024,"candidate_block_size":16}})",
      "t");
  require(ib.engine.index_topk == 2048 && ib.engine.candidate_topk_blocks == 1024 && ib.engine.candidate_block_size == 16,
          "the index budget keys parse");
  // The checkpoint's activation recipe (2026-10-15): "bf16" (the default chain) or "native".
  const auto na = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"prefill_activations":"native"}})", "t");
  require(na.engine.prefill_activations == "native", "engine.prefill_activations native parses");
  // The prefill attention's arithmetic on the fp8 K/V pool (2026-10-16): "bf16" (the default) or "fp8".
  const auto pa = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"kv_dtype":"fp8","prefill_attention":"fp8"}})", "t");
  require(pa.engine.prefill_attention == "fp8" && pa.engine.kv_dtype == "fp8", "engine.prefill_attention fp8 parses");
  bool rejected = false;
  try {
    (void)dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"],"engine":{"prefill_attention":"e4m3"}})", "t");
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "engine.prefill_attention takes bf16 or fp8 only");
  const auto da = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"kv_dtype":"fp8","decode_attention":"fp8"}})", "t");
  require(da.engine.decode_attention == "fp8" && da.engine.prefill_attention == "bf16",
          "native decode is independent of prefill");
  rejected = false;
  try {
    (void)dgpp::serve::parse_cluster_config(
        R"({"model":"m","nodes":["h"],"engine":{"decode_attention":"e4m3"}})", "t");
  } catch (const std::exception&) { rejected = true; }
  require(rejected, "engine.decode_attention takes bf16 or fp8 only");
  // The DFlash2 drafter's keys (2026-10-03).
  const auto df = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"dflash_model":"z-lab/Qwen3.8-27B-DFlash2","dflash_verify_graph":false,"dflash_draft_batch":false,"dflash_depth":4}})", "t");
  require(df.engine.dflash_model == "z-lab/Qwen3.8-27B-DFlash2" && !df.engine.dflash_verify_graph &&
              !df.engine.dflash_draft_batch && df.engine.dflash_depth == 4,
          "the dflash keys parse");
  const auto compact = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"compact_batches":true}})", "t");
  require(compact.engine.compact_batches, "compact_batches enabled");
  const auto fixed = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"compact_batches":false}})", "t");
  require(!fixed.engine.compact_batches, "compact_batches explicit override");
  for (const std::string mode : {"gemv", "mma"}) {
    const auto head = dgpp::serve::parse_cluster_config(
        R"({"model":"m","nodes":["h"],"engine":{"fp8_head":")" + mode + R"("}})", "t");
    require(head.engine.fp8_head == mode, "fp8_head accepts " + mode);
  }
  // The draft head's read (2026-10-13): exact | screen, with the screen's candidate cap.
  for (const std::string mode : {"exact", "screen"}) {
    const auto dh = dgpp::serve::parse_cluster_config(
        R"({"model":"m","nodes":["h"],"engine":{"draft_head":")" + mode + R"(","draft_head_blocks":48}})", "t");
    require(dh.engine.draft_head == mode && dh.engine.draft_head_blocks == 48, "draft_head accepts " + mode);
  }
  // The KV dtype: named by the config, checked by name.
  const dgpp::serve::ClusterConfig fp8 = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"kv_dtype":"fp8"}})", "t");
  require(fp8.engine.kv_dtype == "fp8", "kv_dtype fp8");
  // The Qwen n-gram table's snapshot (the AutoRound hybrid): a repository id,
  // empty by default.
  require(fp8.engine.ngram_table_model.empty(), "ngram_table_model empty by default");
  const dgpp::serve::ClusterConfig table = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"ngram_table_model":"Qwen/Qwen3.8-Flash-Next-FP8"}})", "t");
  require(table.engine.ngram_table_model == "Qwen/Qwen3.8-Flash-Next-FP8", "ngram_table_model parses");
  require(!refusal(R"({"model":"m","nodes":["h"],"engine":{"ngram_table_model":"nope"}})").empty(),
          "ngram_table_model must be ORG/NAME");
  require(fp8.engine.engram_source.empty(), "engram_source empty by default");
  const dgpp::serve::ClusterConfig engram = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"engram_source":"deepseek-ai/DeepSeek-V4.1-Flash-1017"}})", "t");
  require(engram.engine.engram_source == "deepseek-ai/DeepSeek-V4.1-Flash-1017", "engram_source parses");
  require(!refusal(R"({"model":"m","nodes":["h"],"engine":{"engram_source":"nope"}})").empty(),
          "engram_source must be ORG/NAME");
  // The bf16 weights' resident form: named by the config, off by default.
  const dgpp::serve::ClusterConfig bf12 = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"bf16_weights":"bf12"}})", "t");
  require(bf12.engine.bf16_weights == "bf12", "bf16_weights bf12");
  const dgpp::serve::ClusterConfig both = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"bf16_weights":"bf12+bf16"}})", "t");
  require(both.engine.bf16_weights == "bf12+bf16", "bf16_weights bf12+bf16");
  // The full GLM-5.3 draft head's form: named by the config, the checkpoint's by default.
  require(bf12.engine.mtp_head == "checkpoint", "mtp_head defaults to checkpoint");
  const dgpp::serve::ClusterConfig mtph = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"mtp_head":"fp8"}})", "t");
  require(mtph.engine.mtp_head == "fp8", "mtp_head fp8");
  require(!refusal(R"({"model":"m","nodes":["h"],"engine":{"mtp_head":"int8"}})").empty(),
          "mtp_head refuses an unknown form");
  require(bf12.engine.attention_weights == "checkpoint", "attention_weights defaults to checkpoint");
  const dgpp::serve::ClusterConfig attw = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"engine":{"attention_weights":"int4"}})", "t");
  require(attw.engine.attention_weights == "int4", "attention_weights int4");
  require(!refusal(R"({"model":"m","nodes":["h"],"engine":{"attention_weights":"fp8"}})").empty(),
          "attention_weights refuses an unknown form");
  require(c.paths.log_dir == "/var/log/dgpp" && c.paths.stage_dir == "/tmp/bus4" &&
              c.paths.release_dir == "~/dgpp/releases" && c.paths.resident_cache.empty(),
          "the paths: given one taken, the rest defaulted");
  // A one-node config is a world of one.
  const dgpp::serve::ClusterConfig one =
      dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t");
  require(one.world() == 1 && one.http_port == 18080, "a one-node world");
  require(one.http_bind == "0.0.0.0", "HTTP defaults to all IPv4 interfaces");
  require(one.http_max_body_bytes == 256ll * 1024 * 1024, "HTTP body default supports large prefills");
}

DGPP_TEST(cluster_config_stream_flush_defaults_overrides_and_validation) {
  const auto defaults = dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t");
  require(defaults.stream_flush == "pass", "polling remains the default");
  for (const char* mode : {"pass", "wake"}) {
    const auto config = dgpp::serve::parse_cluster_config(
        std::string(R"({"model":"m","nodes":["h"],"http":{"stream_flush":")") + mode + "\"}}", "t");
    require(config.stream_flush == mode, "explicit mode survives parsing");
  }
  for (const char* value : {"0", "true", "null", "[]", "{}", "\"\"", "\"Wake\"", "\"poll\""}) {
    const auto error = refusal(
        std::string(R"({"model":"m","nodes":["h"],"http":{"stream_flush":)") + value + "}}");
    require(error.find("http.stream_flush") != std::string::npos, "invalid mode rejected by name");
  }
}

DGPP_TEST(cluster_config_sse_ping_interval_defaults_overrides_and_validation) {
  const auto defaults = dgpp::serve::parse_cluster_config(R"({"model":"m","nodes":["h"]})", "t");
  require(defaults.sse_ping_interval == 30, "streams ping after 30 seconds by default");
  for (int interval : {-1, 1, 30, 2147483647}) {
    const auto config = dgpp::serve::parse_cluster_config(
        R"({"model":"m","nodes":["h"],"http":{"sse_ping_interval":)" + std::to_string(interval) +
            "}}",
        "t");
    require(config.sse_ping_interval == interval, "explicit interval survives config parsing");
  }
  for (const char* value :
       {"0", "-2", "true", "null", "\"30\"", "1.5", "1.0", "2147483648", "1e100"}) {
    const auto error = refusal(
        std::string(R"({"model":"m","nodes":["h"],"http":{"sse_ping_interval":)") + value + "}}");
    require(error.find("http.sse_ping_interval") != std::string::npos,
            "invalid interval rejected by name");
  }
}

DGPP_TEST(cluster_config_world_size_one_runs_locally) {
  const auto c = dgpp::serve::parse_cluster_config(
      R"({"model":"m","world_size":1})", "t");
  require(c.world() == 1 && c.nodes[0] == "localhost", "single-node shortcut stays local");
}

DGPP_TEST(cluster_config_world_size_requires_resolved_peer_addresses) {
  for (const int world : {2, 4}) {
    const std::string error = refusal(
        R"({"model":"m","world_size":)" + std::to_string(world) + "}");
    require(error.find("explicit 'nodes'") != std::string::npos,
            "multi-node shortcut must require real peer addresses: " + error);
  }
  const auto c = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["head","peer"]})", "t");
  require(c.world() == 2 && c.nodes[0] == "head" && c.nodes[1] == "peer",
          "resolved multi-node configurations preserve the rank addresses");
}

DGPP_TEST(cluster_config_world_size_and_nodes_are_mutually_exclusive) {
  for (const char* json : {
           R"({"model":"m","world_size":2,"nodes":["head","peer"]})",
           R"({"model":"m","nodes":["head","peer"],"world_size":2})",
           R"({"model":"m","world_size":1,"nodes":["head"]})",
           R"({"model":"m","nodes":["head"],"world_size":1})"}) {
    const std::string error = refusal(json);
    require(error.find("'nodes' and 'world_size' are mutually exclusive") != std::string::npos,
            "mixed configuration must fail regardless of key order: " + error);
  }
}

DGPP_TEST(cluster_config_http_override_is_order_independent) {
  const auto c = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"http":{"bind_host":"0.0.0.0","port":8080,"max_body_bytes":5368709120},"ports":{"http":18080},"node_env":[{"HF_HUB_CACHE":"~/cache"}]})", "t");
  require(c.http_bind == "0.0.0.0" && c.http_port == 8080, "deployment HTTP wins over legacy ports");
  require(c.http_max_body_bytes == 5368709120ll, "body limit retains a 64-bit byte count");
  require(c.node_env.at(0).at("HF_HUB_CACHE") == "~/cache", "node cache retained");
  const auto local = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h"],"http":{"bind_host":"127.0.0.1"}})", "t");
  require(local.http_bind == "127.0.0.1", "explicit loopback binding is preserved");
  require(!refusal(R"({"model":"m","nodes":["h"],"http":{"bind_host":"localhost"}})").empty(), "bind must be IPv4");
  require(!refusal(R"({"model":"m","nodes":["h"],"node_env":[{"HF_TOKEN":"x"}]})").empty(), "credentials disallowed");
}

DGPP_TEST(cluster_config_accepts_per_rank_registration_diagnostics) {
  const auto c = dgpp::serve::parse_cluster_config(
      R"({"model":"m","nodes":["h","w"],"node_env":[
        {"DGPP_LOG_LEVEL":"debug","DGPP_MLOCK":"off"},
        {"DGPP_LOG_LEVEL":"info","DGPP_MLOCK":"on"}]})", "t");
  require(c.node_env.at(0).at("DGPP_LOG_LEVEL") == "debug" &&
              c.node_env.at(0).at("DGPP_MLOCK") == "off" &&
              c.node_env.at(1).at("DGPP_LOG_LEVEL") == "info" &&
              c.node_env.at(1).at("DGPP_MLOCK") == "on",
          "both diagnostic settings retain per-rank values");
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
      {R"({"model":"m","nodes":["h"],"engine":{"max_concurency":2}})",
       "unknown key 'engine.max_concurency'"},
      {R"({"model":"m","nodes":["h"],"engine":{"max_concurrency":"2"}})",
       "'engine.max_concurrency' must be an integer"},
      {R"({"model":"m","nodes":["h"],"engine":{"max_concurrency":1.5}})",
       "'engine.max_concurrency' must be an integer"},
      {R"({"model":"m","nodes":["h"],"engine":{"max_concurrency":0}})",
       "'engine.max_concurrency' must be in [1,"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp":"yes"}})",
       "'engine.mtp' must be true or false"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_depth":6}})",
       "'engine.mtp_depth' must be in [1, 5]"},
      {R"({"model":"m","nodes":["h"],"engine":{"dflash_depth":8}})",
       "'engine.dflash_depth' must be in [0, 7]"},
      {R"({"model":"m","nodes":["h"],"engine":{"dflash_verify_graph":"yes"}})",
       "'engine.dflash_verify_graph' must be true or false"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_depth":0}})",
       "'engine.mtp_depth' must be in [1, 5]"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_draft":"beam"}})",
       "'engine.mtp_draft' must be auto, sampled or greedy"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_draft_temperature":0}})",
       "'engine.mtp_draft_temperature' must be in (0, 4]"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_verify":"tree"}})",
       "'engine.mtp_verify' must be token or block"},
      {R"({"model":"m","nodes":["h"],"engine":{"dflash_weights":"nvfp4"}})",
       "'engine.dflash_weights' must be checkpoint or fp8"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_expert_format":"int8"}})",
       "'engine.mtp_expert_format' must be \"fp8\", \"bf16_fused\" or \"bf16\""},
      {R"({"model":"m","nodes":["h"],"engine":{"l2_prefetch_layer_rate":"heavy"}})",
       "'engine.l2_prefetch_layer_rate' must be off, light or full"},
      {R"({"model":"m","nodes":["h"],"engine":{"l2_prefetch_form":"walk"}})",
       "'engine.l2_prefetch_form' must be load, lines or touch"},
      {R"({"model":"m","nodes":["h"],"engine":{"l2_prefetch_full_form":"32x8"}})",
       "'engine.l2_prefetch_full_form' must be 16x8 or 48x4"},
      {R"({"model":"m","nodes":["h"],"engine":{"mtp_schedule_sampled_scale":1.5}})",
       "'engine.mtp_schedule_sampled_scale' must be in [0, 1]"},
      {R"({"model":"m","nodes":["h"],"engine":{"prefill_dense":"fast"}})",
       "'engine.prefill_dense' must be \"auto\", \"tile\" or \"streaming\""},
      {R"({"model":"m","nodes":["h"],"engine":{"prefill_window":"mma"}})",
       "'engine.prefill_window' must be \"flash\" or \"scalar\""},
      {R"({"model":"m","nodes":["h"],"engine":{"decode_window":"split"}})",
       "'engine.decode_window' must be \"flash\" or \"scalar\""},
      {R"({"model":"m","nodes":["h"],"engine":{"collective_transport":"doorbell"}})",
       "'engine.collective_transport' must be \"write\" or \"send\""},
      {R"({"model":"m","nodes":["h"],"engine":{"admission":"fast"}})",
       "'engine.admission' must be \"full\" or \"grow\""},
      {R"({"model":"m","nodes":["h"],"engine":{"prefix_cache_gib":-1}})",
       "'engine.prefix_cache_gib' must be >= 0"},
      {R"({"model":"m","nodes":["h"],"engine":{"prefix_min_tokens":-1}})",
       "'engine.prefix_min_tokens' must be in [0, 1073741824]"},
      {R"({"model":"m","nodes":["h"],"engine":{"prefix_head_snapshots":"yes"}})",
       "'engine.prefix_head_snapshots' must be true or false"},
      {R"({"model":"m","nodes":["h"],"engine":{"kv_dtype":"int8"}})",
       "'engine.kv_dtype' must be \"bf16\", \"fp8\" or \"fp4\""},
      {R"({"model":"m","nodes":["h"],"engine":{"kv_dtype":8}})",
       "'engine.kv_dtype' must be a string"},
      {R"({"model":"m","nodes":["h"],"engine":{"prefill":"fast"}})",
       "'engine.prefill' must be \"bounded\" or \"exact\""},
      {R"({"model":"m","nodes":["h"],"engine":{"compact_batches":"true"}})",
       "'engine.compact_batches' must be true or false"},
      {R"({"model":"m","nodes":["h"],"engine":{"prefill_fold_scales":1}})",
       "'engine.prefill_fold_scales' must be true or false"},
      {R"({"model":"m","nodes":["h"],"engine":{"prefill_fp8_gemm":"on"}})",
       "'engine.prefill_fp8_gemm' must be true or false"},
      {R"({"model":"m","nodes":["h"],"engine":{"expert_gemm":"fast"}})",
       "'engine.expert_gemm' must be \"wide\", \"wide3\", \"wide4\", \"wide4r\" or \"narrow\""},
      {R"({"model":"m","nodes":["h"],"engine":{"expert_tile_list":"yes"}})",
       "'engine.expert_tile_list' must be true or false"},
      {R"({"model":"m","nodes":["h"],"engine":{"fp8_head":"auto"}})",
       "'engine.fp8_head' must be \"gemv\" or \"mma\""},
      {R"({"model":"m","nodes":["h"],"engine":{"fp8_head":true}})",
       "'engine.fp8_head' must be a string"},
      {R"({"model":"m","nodes":["h"],"engine":{"draft_head":"auto"}})",
       "'engine.draft_head' must be \"exact\" or \"screen\""},
      {R"({"model":"m","nodes":["h"],"engine":{"draft_head_blocks":0}})",
       "'engine.draft_head_blocks' must be in [1, 64]"},
      {R"({"model":"m","nodes":["h"],"engine":{"index_topk":4096}})",
       "'engine.index_topk' must be in [0, 2048]"},
      {R"({"model":"m","nodes":["h"],"engine":{"index_topk":1000}})",
       "'engine.index_topk' must be 0 (the checkpoint's) or a power of two <= 2048"},
      {R"({"model":"m","nodes":["h"],"engine":{"candidate_topk_blocks":3}})",
       "'engine.candidate_topk_blocks' must be 0 (the checkpoint's) or a power of two <= 2048"},
      {R"({"model":"m","nodes":["h"],"engine":{"candidate_block_size":-1}})",
       "'engine.candidate_block_size' must be in [0, 1024]"},
      {R"({"model":"m","nodes":["h"],"engine":{"bf16_weights":"fp8"}})",
       "'engine.bf16_weights' must be \"checkpoint\", \"bf12\" or \"bf12+bf16\""},
      {R"({"model":"m","nodes":["h"],"engine":{"bf16_weights":true}})",
       "'engine.bf16_weights' must be a string"},
      {R"({"model":"m","nodes":["h"],"ports":{"http":70000}})",
       "'ports.http' must be in [1, 65535]"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":0}})",
       "'http.max_body_bytes' must be in [1,"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":-1}})",
       "'http.max_body_bytes' must be in [1,"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":true}})",
       "'http.max_body_bytes' must be an integer"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":"256MiB"}})",
       "'http.max_body_bytes' must be an integer"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":1.5}})",
       "'http.max_body_bytes' must be an integer"},
      {R"({"model":"m","nodes":["h"],"http":{"max_body_bytes":9223372036854775808}})",
       "http.max_body_bytes"},
      {R"({"model":"m","nodes":["h"],"ports":{"fabric":5,"journal":5}})",
       "'ports.fabric' and 'ports.journal' must differ"},
      {R"({"model":"m","nodes":["h"],"ports":{"metrics":70000}})",
       "'ports.metrics' must be in [0, 65535]"},
      {R"({"model":"m","nodes":["h"],"ports":{"metrics":29970}})",
       "'ports.metrics' must differ from 'ports.fabric' and 'ports.journal'"},
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
              ex.engine.kv_dtype == "bf16" && ex.engine.bf16_weights == "bf12+bf16" &&
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

DGPP_TEST(cluster_config_the_fp4local_template_is_a_deployment_the_engine_reads) {
  // deploy/cluster_qwen-3.8-flash-next-fp4local_w1.example.json, the
  // ready-to-run shape for the FP4-Local export (2026-10-17). The resident
  // footprint (98.2 GiB: the experts with their CSF-wrapped scales and the
  // resident n-gram table's codes) fits one 121.6 GiB device, so the
  // template seats a whole model in world 1. What this checks: every field
  // name in the file is one this parser reads (an unknown one is a hard
  // failure), and the recipe the loader was validated against — the MTP
  // draft at depth 3 with the screened head behind the lookup draft, the
  // BF16 remainder FP8 at load, the table mapped rather than resident, the
  // Full layers' K/V at fp8 with the fp8 prefill attention (half the bf16
  // pool, sized here to four full 262144-token contexts; interleaved on one
  // Spark it measured 1459-1469 t/s prefill at 4-8K depth against 1141-1260
  // bf16, decode unchanged, 2026-10-18), and the YaRN ramp to a 1M positional
  // ceiling (the cap, not the pool) — is what the file spells.
  const std::string path = std::string(DGPP_SOURCE_DIR) +
                           "/deploy/cluster_qwen-3.8-flash-next-fp4local_w1.example.json";
  std::ifstream in(path);
  require(in.good(), "the tracked template is readable: " + path);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const dgpp::serve::ClusterConfig c = dgpp::serve::parse_cluster_config(text, path);
  require(c.model == "nvidia/Qwen3.8-Flash-Next-FP4-Local" && c.world() == 1,
          "the single-Spark FP4-Local deployment");
  require(c.engine.mtp && c.engine.mtp_depth == 3 && c.engine.decode_graph &&
              c.engine.draft_head == "screen" && c.engine.lookup_draft && c.engine.lookup_tail == 2 &&
              c.engine.ngram_table == "mmap",
          "the draft and lookup recipe the validation measured");
  require(c.engine.dense_weights == "fp8" && c.engine.fp8_head == "mma",
          "the BF16 remainder FP8 at load and the tensor-core head, as the sibling NVFP4 templates");
  require(c.engine.kv_dtype == "fp8" && c.engine.prefill_attention == "fp8",
          "the Full layers' K/V at fp8 with the fp8 prefill attention: half the bf16 pool, the long-depth prefill win");
  require(c.engine.rope_scaling.has_value() && c.engine.rope_scaling->factor == 4.0 &&
              c.engine.rope_scaling->original_max_position_embeddings == 262144 &&
              c.engine.rope_scaling->context_limit() == 1048576,
          "the YaRN ramp to a 1M positional ceiling: the cap moves, not the pool (it costs no K/V bytes)");
  require(c.engine.max_concurrency == 4 && c.engine.kv_capacity % 64 == 0 &&
              c.engine.kv_capacity > c.engine.default_max_tokens && c.engine.prefix_cache_gib == 3 &&
              c.engine.model_alias == "Qwen3.8-Flash-Next-FP4",
          "four request slots in a pool that clears a maximal answer, and the one-node Qwen prefix arena");
}

DGPP_TEST(cluster_config_prefill_defaults_to_auto_with_explicit_opt_out) {
  const auto defaults = dgpp::serve::parse_cluster_config(engine_json("{}"), "t");
  require(defaults.engine.prefill_budget_tokens == -1, "unset prefill budget is automatic");
  for (int budget : {-1, 0, 256}) {
    const auto config = dgpp::serve::parse_cluster_config(
        engine_json("{\"prefill_budget_tokens\":" + std::to_string(budget) + "}"), "t");
    require(config.engine.prefill_budget_tokens == budget, "explicit budget preserved");
  }
}

DGPP_TEST(cluster_config_latency_targets_have_serving_defaults_and_strict_overrides) {
  const auto defaults = dgpp::serve::parse_cluster_config(engine_json("{}"), "t").engine;
  require(defaults.decode_gap_target_ms == 1000 && defaults.first_token_target_ms == 3000,
          "model-independent serving latency defaults");
  const auto fixed = dgpp::serve::parse_cluster_config(
      engine_json(R"({"decode_gap_target_ms":0,"first_token_target_ms":0})"), "t").engine;
  require(fixed.decode_gap_target_ms == 0 && fixed.first_token_target_ms == 0, "fixed-budget control remains configurable");
  const auto custom = dgpp::serve::parse_cluster_config(
      engine_json(R"({"decode_gap_target_ms":300,"first_token_target_ms":1500})"), "t").engine;
  require(custom.decode_gap_target_ms == 300 && custom.first_token_target_ms == 1500, "deployment targets survive parsing");
  for (const std::string key : {"decode_gap_target_ms", "first_token_target_ms"})
    for (const std::string value : {"-1", "1.5", "3600001", "true", "\"500\"", "null"})
      require(refusal(engine_json("{\"" + key + "\":" + value + "}")).find(key) != std::string::npos,
              "bad latency target rejected by name");
}

DGPP_TEST(cluster_config_latency_tuning_defaults_and_bounds) {
  const auto defaults = dgpp::serve::parse_cluster_config(engine_json("{}"), "t").engine.prefill_latency;
  require(defaults == dgpp::sched::PrefillLatencyTuning{15, 2, 25}, "stable controller defaults");
  const auto custom = dgpp::serve::parse_cluster_config(engine_json(
      R"({"prefill_latency_headroom_pct":20,"prefill_growth_factor":4,"prefill_cost_recovery_pct":50,"prefill_initial_tokens":512,"prefill_idle_floor_tokens":2048})"), "t");
  require(custom.engine.prefill_latency == dgpp::sched::PrefillLatencyTuning{20, 4, 50, 512, 2048}, "custom controller tuning survives parsing");
  for (const std::string key : {"prefill_latency_headroom_pct", "prefill_growth_factor", "prefill_cost_recovery_pct"})
    for (const std::string value : {"-1", "101", "1.5", "true", "null"})
      require(refusal(engine_json("{\"" + key + "\":" + value + "}")).find(key) != std::string::npos,
              "invalid controller tuning rejected by name");
  for (const std::string value : {"-1", "0", "1073741825", "1.5", "true", "null"})
    require(refusal(engine_json("{\"prefill_initial_tokens\":" + value + "}")).find("prefill_initial_tokens") != std::string::npos,
            "invalid initial budget rejected by name");
  for (const std::string value : {"-1", "1073741825", "1.5", "true", "null"})
    require(refusal(engine_json("{\"prefill_idle_floor_tokens\":" + value + "}")).find("prefill_idle_floor_tokens") != std::string::npos,
            "invalid idle floor rejected by name");
  require(dgpp::serve::parse_cluster_config(engine_json(R"({"prefill_idle_floor_tokens":0})"), "t").engine.prefill_latency.idle_floor_tokens == 0,
          "a zero floor (the alignment) is a valid choice");
}

DGPP_TEST(cluster_config_tokenizer_cache) {
  const auto parse = [](const std::string& tokenizer) {
    return dgpp::serve::parse_cluster_config(
        "{\"model\":\"m\",\"world_size\":1,\"tokenizer\":" + tokenizer + "}", "t");
  };
  require(parse("{}").tokenizer_word_cache_entries == 0, "cache disabled by default");
  require(parse("{\"word_cache_entries\":10000}").tokenizer_word_cache_entries == 10000,
          "cache capacity parsed");
  for (const char* value : {"-1", "10001", "1.5", "1.0", "true", "null", "\"10\""}) {
    const std::string json =
        "{\"model\":\"m\",\"world_size\":1,\"tokenizer\":{\"word_cache_entries\":" +
        std::string(value) + "}}";
    require(refusal(json).find("tokenizer.word_cache_entries") != std::string::npos,
            "invalid cache capacity must name the field");
  }
  for (const char* value : {"null", "[]", "{\"unknown\":1}"}) {
    const std::string json =
        "{\"model\":\"m\",\"world_size\":1,\"tokenizer\":" + std::string(value) + "}";
    require(refusal(json).find("tokenizer") != std::string::npos,
            "malformed tokenizer config refused");
  }
}

DGPP_TEST(cluster_config_prefill_dense_defaults_to_family_policy) {
  const auto a = dgpp::serve::parse_cluster_config(engine_json("{}"), "test");
  require(a.engine.prefill_dense == "auto", "automatic family policy is the default");
  const auto b = dgpp::serve::parse_cluster_config(engine_json(R"({"prefill_dense":"auto"})"), "test");
  require(b.engine.prefill_dense == "auto", "explicit automatic policy is accepted");
}
