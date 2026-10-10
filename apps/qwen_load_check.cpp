// qwen_load_check: loads a Qwen3.8-Flash-Next checkpoint through the
// resident loader at one rank's TP geometry and reports what the Q2 gate
// asks for — per-layer bytes and times, the byte formulas against actual
// usage, the source-byte plan against the bytes read, the n-gram table
// slice, and the replicated digest (compare across ranks by hand or by
// the fabric script). No bus, no forward: one process per rank, run on
// each node independently.
//
//   qwen_load_check --model ORG/NAME | --checkpoint-dir DIR
//                   [--world W] [--rank R] [--streaming] [--mtp]
//                   [--layers N] [--threads T] [--image-dir DIR|off]
// threads > 1 loads the range on a worker pool (resident mode only;
// per-layer lines are skipped, the summary still reports totals).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "common/log.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/loader.hpp"

int main(int argc, char** argv) {
  std::string model_id, ckpt, image_dir, ngram_table, dense_weights, mtp_expert_format;
  std::string ngram_table_dir;
  int world = 1, rank = 0, layers = -1, threads = 1;
  bool streaming = false, mtp = false;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--world") world = std::stoi(next(i));
      else if (a == "--rank") rank = std::stoi(next(i));
      else if (a == "--streaming") streaming = true;
      else if (a == "--mtp") mtp = true;
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--threads") threads = std::stoi(next(i));
      else if (a == "--image-dir") image_dir = next(i);
      else if (a == "--ngram-table") ngram_table = next(i);
      else if (a == "--ngram-table-dir") ngram_table_dir = next(i);
      else if (a == "--dense-weights") dense_weights = next(i);
      else if (a == "--mtp-expert-format") mtp_expert_format = next(i);
      else throw std::runtime_error("unknown argument " + a);
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    if (dgpp::detect_architecture_file(cfg_path) != dgpp::ModelArchitecture::Qwen4Exp)
      throw std::runtime_error("not a Qwen4Exp checkpoint: " + ckpt);
    dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file(cfg_path);
    if (!image_dir.empty()) dgpp::QwenLayerStream::set_resident_image_dir(image_dir == "off" ? "" : image_dir);
    if (!ngram_table.empty()) dgpp::QwenLayerStream::set_ngram_table_mmap(ngram_table == "mmap");
    if (!ngram_table_dir.empty()) dgpp::QwenLayerStream::set_ngram_table_dir(ngram_table_dir);
    if (cfg.dense_fp8_shipped) dgpp::QwenLayerStream::set_dense_weights_fp8(true);  // the hybrid ships fp8
    if (!dense_weights.empty()) dgpp::QwenLayerStream::set_dense_weights_fp8(dense_weights == "fp8");
    if (!mtp_expert_format.empty()) dgpp::QwenLayerStream::set_mtp_expert_format(mtp_expert_format == "bf16_fused");
    cfg.mtp_experts_bf16_fused = dgpp::QwenLayerStream::mtp_experts_bf16_fused();
    const dgpp::QwenResidency residency = streaming ? dgpp::QwenResidency::Streaming : dgpp::QwenResidency::Resident;
    const dgpp::QwenHeadSharding head = world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full;
    const double kGiB = 1024.0 * 1024.0 * 1024.0;
    DGPP_LOG_INFO("qwen_load_check: {} world {} rank {} {} — formulas: resident {:.2f} GiB (globals {:.2f}, table {:.2f})",
                  ckpt, world, rank, streaming ? "streaming" : "resident",
                  dgpp::QwenLayerStream::resident_bytes(cfg, rank, world, head, mtp) / kGiB,
                  dgpp::QwenLayerStream::globals_bytes(cfg, rank, world, head) / kGiB,
                  dgpp::QwenLayerStream::ngram_table_bytes(cfg, rank, world) / kGiB);
    const auto t0 = std::chrono::steady_clock::now();
    dgpp::QwenLayerStream stream(cfg, ckpt, rank, world, residency, head, mtp);
    const auto t1 = std::chrono::steady_clock::now();
    DGPP_LOG_INFO("qwen_load_check: opened in {:.1f} s; layer capacity {:.2f} GiB", std::chrono::duration<double>(t1 - t0).count(), stream.layer_capacity() / kGiB);
    const dgpp::QwenReplicatedDigest d = stream.hash_replicated();
    DGPP_LOG_INFO("qwen_load_check: digest globals {:016x} tensors {} bytes {:.2f} GiB; layer 0 {:016x} layer 1 {:016x} last {:016x}",
                  d.globals, d.tensors, d.bytes / kGiB, d.layer[0], d.layer[1], d.layer.back());
    const auto& g = stream.load_globals();
    DGPP_LOG_INFO("qwen_load_check: globals {:.2f} GiB, lm head rows [{}, +{})", g.bytes / kGiB, g.lm_vocab_begin, g.lm_vocab_count);
    const auto t2 = std::chrono::steady_clock::now();
    const auto& tbl = stream.load_ngram_table();
    DGPP_LOG_INFO("qwen_load_check: n-gram table rows [{}, +{}) {:.2f} GiB in {:.1f} s", tbl.row_begin, tbl.rows, tbl.bytes / kGiB, std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count());
    if (tbl.mmap) {
      // The mmap'ed table's gather cost: a decode pass's rows
      // (2 tokens x 16 heads, uniformly random rows — no locality, the
      // worst case) and a prefill chunk's (2048 x 16), timed on the host
      // as the walk's host node runs them.
      const dgpp::QwenNgramGeometry ng = cfg.ngram_geometry();
      const int heads = ng.heads;
      std::vector<int32_t> ids;
      std::vector<uint8_t> dst;
      uint64_t x = 0x9E3779B97F4A7C15ull;
      const auto fill = [&](int rows) {
        ids.resize(static_cast<size_t>(rows) * heads);
        dst.resize(static_cast<size_t>(rows) * heads * ng.head_dim);
        for (int t = 0; t < rows; ++t)
          for (int h = 0; h < heads; ++h) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            ids[static_cast<size_t>(t) * heads + h] = static_cast<int32_t>(ng.head_offset[h] + static_cast<int64_t>(x % static_cast<uint64_t>(ng.head_vocab[h])));
          }
      };
      std::vector<double> us;
      for (int i = 0; i < 200; ++i) {
        fill(2);
        const auto a = std::chrono::steady_clock::now();
        tbl.mmap->gather(ids.data(), 2, heads, 0, heads, dst.data());
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
      }
      std::sort(us.begin(), us.end());
      DGPP_LOG_INFO("qwen_load_check: mmap gather of a 2-row decode pass (32 random rows): p50 {:.0f} us, p90 {:.0f}, p99 {:.0f}, max {:.0f}",
                    us[100], us[180], us[198], us.back());
      us.clear();
      for (int i = 0; i < 4; ++i) {
        fill(2048);
        const auto a = std::chrono::steady_clock::now();
        tbl.mmap->gather(ids.data(), 2048, heads, 0, heads, dst.data());
        us.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
      }
      DGPP_LOG_INFO("qwen_load_check: mmap gather of a 2048-row prefill chunk (32768 random rows): {:.0f} / {:.0f} / {:.0f} / {:.0f} ms",
                    us[0], us[1], us[2], us[3]);
    }
    const int n = layers < 0 ? cfg.num_hidden_layers + (mtp && cfg.mtp_layer() >= 0 ? 1 : 0) : layers;
    size_t total = 0;
    const auto t_layers = std::chrono::steady_clock::now();
    if (threads > 1 && !streaming) {
      stream.load_layers_parallel(0, n, threads);
      for (int l = 0; l < n; ++l) total += stream.layer_bytes(cfg, l, rank, world);
    } else {
      for (int l = 0; l < n; ++l) {
        const auto tl = std::chrono::steady_clock::now();
        const uint64_t before = stream.source_bytes_read();
        const auto& r = stream.load_layer(l);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count();
        total += r.bytes;
        if (l < 3 || l == n - 1 || l % 12 == 0)
          DGPP_LOG_INFO("qwen_load_check: layer {} ({}{}) {:.3f} GiB in {:.2f} s, read {:.3f} GiB; experts {} x {} (scale block {}){}",
                        l, r.kind == dgpp::QwenLayerKind::Gdn ? "gdn" : "qsa", r.has_ple ? "+ple" : "",
                        r.bytes / kGiB, s, (stream.source_bytes_read() - before) / kGiB,
                        r.moe.experts.size() / 3, r.moe.local_inter, r.moe.scale_block,
                        r.has_ple ? std::format(", ple heads [{}, +{}) rows [{}, +{})", r.ple.hash_head_begin, r.ple.hash_heads, r.ple.row_begin, r.ple.rows) : "");
        if (!streaming) continue;
        stream.release_layer();
      }
    }
    DGPP_LOG_INFO("qwen_load_check: layer range loaded in {:.1f} s on {} thread(s)",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t_layers).count(),
                  threads > 1 && !streaming ? threads : 1);
    if (!streaming) stream.release_sources();
    DGPP_LOG_INFO("qwen_load_check: {} layers {:.2f} GiB resident (+ globals {:.2f} + table {:.2f} = {:.2f} GiB); source bytes read {:.2f} GiB (verbatim {:.2f}); image restored {} captured {}; total {:.1f} s",
                  n, total / kGiB, g.bytes / kGiB, tbl.bytes / kGiB, (total + g.bytes + tbl.bytes) / kGiB,
                  stream.source_bytes_read() / kGiB, stream.verbatim_source_bytes() / kGiB,
                  stream.image_layers_restored(), stream.image_layers_captured(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("qwen_load_check: {}", e.what());
    return 1;
  }
}
