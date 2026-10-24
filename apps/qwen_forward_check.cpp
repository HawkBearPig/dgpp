// qwen_forward_check: the Qwen3.8-Flash-Next world-1 diagnostic forward
// over a real checkpoint — one prompt of token ids
// through the streaming resident loader, the per-layer hyper-state
// magnitudes and every position's top-k next-token logits. No tokenizer
// yet (Q5): ids come in on the command line.
//
//   qwen_forward_check --model ORG/NAME | --checkpoint-dir DIR
//                      --ids 1,2,3,... [--topk K] [--layers N] [--dump-states FILE]
//                      [--world W --rank R --peer HOST --port N] [--resident]
//                      [--image-dir DIR|off] [--lat-slot-bytes N]
//                      [--kv-dtype bf16|fp8] [--prefill-attention bf16|fp8]
//                      [--dump-logits FILE] (row-major fp32, this rank's vocabulary)
//
// TP (plan D1): one process per node over the fabric bus; every fold of
// the diagnostic forward rides the latency path, so the slot is sized for
// the widest one (the PLE layer's [T, hc*hidden] key partial) — the bulk
// machine stalls at world 4 on buffers spanning fewer stripes than ranks
// (2026-09-09, the plan's risks). Each rank prints its per-layer hyper
// state digests (bitwise identical across ranks by contract) and the
// merged top-1 needs every rank's vocab slice: each rank prints its own
// slice's argmax with the logit, the script merges.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "engine/graph_engine.hpp"
#include "engine/tp_bus.hpp"
#include "loaders/architecture.hpp"
#include "net/collective_bus.hpp"
#include "loaders/hf_cache.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/forward.hpp"

int main(int argc, char** argv) {
  std::string model_id, ckpt, ids_text, dump_states, peer, image_dir, ngram_table, dense_weights, mtp_expert_format;
  std::string ngram_table_dir, dump_logits, kv_dtype = "bf16", prefill_attention = "bf16";
  int topk = 5, layers = -1, world = 1, rank = 0, port = 29950;
  bool resident = false;
  size_t lat_slot_bytes = 0;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--ids") ids_text = next(i);
      else if (a == "--topk") topk = std::stoi(next(i));
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--dump-states") dump_states = next(i);
      else if (a == "--dump-logits") dump_logits = next(i);
      else if (a == "--kv-dtype") kv_dtype = next(i);
      else if (a == "--prefill-attention") prefill_attention = next(i);
      else if (a == "--world") world = std::stoi(next(i));
      else if (a == "--rank") rank = std::stoi(next(i));
      else if (a == "--peer") peer = next(i);
      else if (a == "--port") port = std::stoi(next(i));
      else if (a == "--resident") resident = true;
      else if (a == "--image-dir") image_dir = next(i);
      else if (a == "--ngram-table") ngram_table = next(i);
      else if (a == "--ngram-table-dir") ngram_table_dir = next(i);
      else if (a == "--dense-weights") dense_weights = next(i);
      else if (a == "--mtp-expert-format") mtp_expert_format = next(i);
      else if (a == "--lat-slot-bytes") lat_slot_bytes = std::stoull(next(i));
      else throw std::runtime_error("unknown argument " + a);
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    if (ids_text.empty()) throw std::runtime_error("--ids is required");
    if ((kv_dtype != "bf16" && kv_dtype != "fp8") ||
        (prefill_attention != "bf16" && prefill_attention != "fp8"))
      throw std::runtime_error("--kv-dtype and --prefill-attention must be bf16 or fp8");
    if (prefill_attention == "fp8" && kv_dtype != "fp8")
      throw std::runtime_error("--prefill-attention fp8 requires --kv-dtype fp8");
    dgpp::QwenModel::set_native_attention(prefill_attention == "fp8", false);
    std::vector<int64_t> ids;
    {
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::stoll(item));
    }
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    if (dgpp::detect_architecture_file(cfg_path) != dgpp::ModelArchitecture::Qwen4Exp)
      throw std::runtime_error("not a Qwen4Exp checkpoint: " + ckpt);
    dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file(cfg_path);
    if (layers > 0 && layers < cfg.num_hidden_layers) {
      cfg.num_hidden_layers = layers;
      cfg.layers.resize(static_cast<size_t>(layers));
    }
    const int T = static_cast<int>(ids.size());
    const int H = cfg.hidden_size, W = cfg.hc_count * H;
    if (!image_dir.empty()) dgpp::QwenLayerStream::set_resident_image_dir(image_dir == "off" ? "" : image_dir);
    if (!ngram_table.empty()) dgpp::QwenLayerStream::set_ngram_table_mmap(ngram_table == "mmap");
    if (!ngram_table_dir.empty()) dgpp::QwenLayerStream::set_ngram_table_dir(ngram_table_dir);
    if (cfg.dense_fp8_shipped) dgpp::QwenLayerStream::set_dense_weights_fp8(true);  // the hybrid ships fp8
    if (!dense_weights.empty()) dgpp::QwenLayerStream::set_dense_weights_fp8(dense_weights == "fp8");
    if (!mtp_expert_format.empty()) dgpp::QwenLayerStream::set_mtp_expert_format(mtp_expert_format == "bf16_fused");
    cfg.mtp_experts_bf16_fused = dgpp::QwenLayerStream::mtp_experts_bf16_fused();
    const dgpp::QwenResidency residency = resident ? dgpp::QwenResidency::Resident : dgpp::QwenResidency::Streaming;
    DGPP_LOG_INFO("qwen_forward_check: {} — {} tokens, {} layers, {} world {} rank {}", ckpt, T,
                  cfg.num_hidden_layers, resident ? "resident" : "streaming", world, rank);
    std::unique_ptr<dgpp::net::CollectiveBus> bus;
    std::unique_ptr<dgpp::BusBoundaryReducer> reducer;
    if (world > 1) {
      if (peer.empty() && rank != 0) throw std::runtime_error("--peer is required on ranks > 0");
      // The widest fold: the PLE layer's [T, hc*hidden] key partial (bf16).
      if (lat_slot_bytes == 0) lat_slot_bytes = static_cast<size_t>(T) * W * 2 + 4096;
      bus = std::make_unique<dgpp::net::CollectiveBus>(
          dgpp::fabric_bus_options(rank, world, static_cast<uint16_t>(port), peer, 120000, lat_slot_bytes));
      std::string err;
      if (!bus->start(&err)) throw std::runtime_error("rank " + std::to_string(rank) + " bus start: " + err);
      reducer = std::make_unique<dgpp::BusBoundaryReducer>(*bus, 600000);
    }
    const auto t0 = std::chrono::steady_clock::now();
    dgpp::QwenModel model(cfg, ckpt, T, T + 64, residency, reducer.get(), rank, world,
                         1, false, 0, false, false,
                         kv_dtype == "fp8" ? dgpp::LatentFormat::kFp8 : dgpp::LatentFormat::kBf16);
    const auto t1 = std::chrono::steady_clock::now();
    const dgpp::QwenModel::Outputs out = model.forward(ids, true);
    const auto t2 = std::chrono::steady_clock::now();
    if (!dump_logits.empty()) {
      std::ofstream f(dump_logits, std::ios::binary);
      f.write(reinterpret_cast<const char*>(out.logits.data()),
              static_cast<std::streamsize>(out.logits.size() * sizeof(float)));
      if (!f) throw std::runtime_error("cannot write logits to " + dump_logits);
      std::printf("wrote %s: [%d, %d] fp32 logits (vocab begins at %d)\n",
                  dump_logits.c_str(), T, out.lm_vocab_count, out.lm_vocab_begin);
    }
    auto fnv = [](const std::vector<uint16_t>& v) {
      uint64_t h = 1469598103934665603ull;
      for (uint16_t x : v) { h ^= x; h *= 1099511628211ull; }
      return h;
    };
    for (size_t l = 0; l < out.layer_states.size(); ++l) {
      double rms = 0, mx = 0;
      for (uint16_t v : out.layer_states[l]) {
        const float f = dgpp::bf16_bits_to_float(v);
        rms += static_cast<double>(f) * f;
        mx = std::max(mx, static_cast<double>(std::fabs(f)));
      }
      rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
      std::printf("layer %2zu: R rms %.4g max %.4g digest %016llx\n", l, rms, mx,
                  static_cast<unsigned long long>(fnv(out.layer_states[l])));
    }
    std::printf("final hidden digest %016llx\n", static_cast<unsigned long long>(fnv(out.final_hidden_bits)));
    // This rank's vocab slice: the top-k with GLOBAL ids and the logits.
    const auto top = dgpp::QwenModel::topk(out.logits, T, out.lm_vocab_count, topk);
    for (int t = 0; t < T; ++t) {
      std::printf("pos %3d id %7lld ->", t, static_cast<long long>(ids[static_cast<size_t>(t)]));
      for (const auto& [id, v] : top[static_cast<size_t>(t)]) std::printf(" %d(%.3f)", id + out.lm_vocab_begin, v);
      std::printf("\n");
    }
    std::printf("argmax ids:");
    for (int t = 0; t < T; ++t) std::printf("%s%d", t ? "," : " ", top[static_cast<size_t>(t)][0].first + out.lm_vocab_begin);
    std::printf("\n");
    std::printf("argmax logits:");
    for (int t = 0; t < T; ++t) std::printf("%s%.4f", t ? "," : " ", top[static_cast<size_t>(t)][0].second);
    std::printf("\n");
    if (!dump_states.empty()) {
      std::ofstream f(dump_states, std::ios::binary);
      for (const auto& st : out.layer_states) f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
      f.write(reinterpret_cast<const char*>(out.final_hidden_bits.data()), static_cast<std::streamsize>(out.final_hidden_bits.size() * 2));
      std::printf("wrote %s: %zu layers x [%d, %d] + final [%d, %d] bf16\n", dump_states.c_str(), out.layer_states.size(), T, W, T, H);
    }
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("boot %.0f ms, forward %.0f ms\n", ms(t0, t1), ms(t1, t2));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen_forward_check: %s\n", e.what());
    return 1;
  }
}
