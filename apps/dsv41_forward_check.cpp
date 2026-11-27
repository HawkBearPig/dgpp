// dsv41_forward_check: the engine's per-layer residual streams and collapse
// coefficients over a prompt on the REAL DeepSeek-V4.1-Flash checkpoint
// (world 1, the streaming loader: each layer read from the NVMe as the walk
// reaches it), dumped for the independent torch reference
// (tools/dsv41_torch_reference.py: the release's own model.py layer code)
// — docs/deepseek_v41_flash_plan.md G4/G5, the cross-check on the real
// weights.
//
//   dsv41_forward_check --model ORG/NAME | --checkpoint-dir DIR --out FILE
//                       (--ids 1,2,3 | --text FILE [--no-bos] | --tokens N [--seed S])
//                       [--layers N] [--bounded] [--act-native]
//
// --act-native: the release's activation recipe on the walk's rows
// (engine.prefill_activations native), the engine's side of the torch
// reference's --act-quant.
//
// The dump: "DSV41ST3", int32 L T H4, bf16 states [L][T][H4] (every walked
// layer's output streams, [4, H] per row), fp32 pre [L][T][4] (the collapse
// coefficients the NEXT layer's attention site uses — the layer's ffn_pre),
// int64 ids [T], then int32 K and the routed expert ids [L][T][K] (ascending),
// int32 Li and ms and the index sources' selections [Li][T][ms] (-1 padded;
// the sources among the walked layers). With every layer walked the last
// row's top-8 logits print.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/log.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/model.hpp"
#include "text/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string model_id, ckpt, out, ids_text, text_path;
  int tokens = 0, layers = 0;
  uint64_t seed = 7;
  bool bos = true, bounded = false, act_native = false;
  const auto next = [&](int& i) {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--out") out = next(i);
      else if (a == "--ids") ids_text = next(i);
      else if (a == "--text") text_path = next(i);
      else if (a == "--tokens") tokens = std::stoi(next(i));
      else if (a == "--seed") seed = std::stoull(next(i));
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--no-bos") bos = false;
      else if (a == "--bounded") bounded = true;
      else if (a == "--act-native") act_native = true;
      else throw std::runtime_error("unknown argument " + a);
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    if (out.empty()) throw std::runtime_error("--out is required");
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    if (dgpp::detect_architecture_file(cfg_path) != dgpp::ModelArchitecture::DeepseekV41)
      throw std::runtime_error("not a DeepseekV41 checkpoint: " + ckpt);
    const dgpp::Dsv41TextConfig cfg = dgpp::Dsv41TextConfig::from_json_file(cfg_path);
    // The prompt: explicit ids, a text through the checkpoint's tokenizer
    // (BOS first unless --no-bos), or random ids.
    std::vector<int64_t> ids;
    if (!ids_text.empty()) {
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) ids.push_back(std::stoll(item));
    } else if (!text_path.empty()) {
      std::ifstream f(text_path);
      if (!f) throw std::runtime_error("cannot open " + text_path);
      std::stringstream buf;
      buf << f.rdbuf();
      const dgpp::text::Tokenizer tok =
          dgpp::text::Tokenizer::load((std::filesystem::path(ckpt) / "tokenizer.json").string());
      if (bos) {
        int64_t bos_id = -1;
        for (const auto& added : tok.added_tokens())
          if (added.content == "<｜begin▁of▁sentence｜>") bos_id = added.id;
        if (bos_id < 0) throw std::runtime_error("the tokenizer has no BOS token");
        ids.push_back(bos_id);
      }
      const std::vector<int64_t> body = tok.encode(buf.str());
      ids.insert(ids.end(), body.begin(), body.end());
      if (tokens > 0 && static_cast<int>(ids.size()) > tokens) ids.resize(static_cast<size_t>(tokens));
    } else {
      if (tokens <= 0) throw std::runtime_error("--ids, --text or --tokens is required");
      std::mt19937_64 rng(seed);
      for (int i = 0; i < tokens; ++i) ids.push_back(static_cast<int64_t>(rng() % static_cast<uint64_t>(cfg.vocab_size)));
    }
    const int T = static_cast<int>(ids.size());
    if (T < 1) throw std::runtime_error("an empty prompt");
    DGPP_LOG_INFO("dsv41_forward_check: {} tokens{}, {} layers, {} prefill, {} activations", T, bounded ? "" : "",
                  layers > 0 ? layers : cfg.num_hidden_layers, bounded ? "bounded" : "exact",
                  act_native ? "the release's fp8 block" : "bf16");
    const auto t0 = std::chrono::steady_clock::now();
    dgpp::Dsv41Model model(cfg, ckpt, /*max_tokens=*/std::max({T, cfg.sliding_window, 16}), /*max_cache_tokens=*/std::max(T + 256, 1024),
                           dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1);
    model.set_prefill_bounded(bounded);
    model.set_prefill_act_native(act_native);
    if (layers > 0) model.set_debug_layer_limit(layers);
    const dgpp::Dsv41Model::Outputs o = model.forward(ids, /*capture_layers=*/true);
    const std::vector<std::vector<float>> pre = model.debug_layer_pre();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int L = static_cast<int>(o.layer_states.size());
    const int H4 = cfg.hc_mult * cfg.hidden_size;
    if (L < 1 || static_cast<int>(pre.size()) != L) throw std::runtime_error("no layer captures");
    std::ofstream f(out, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + out);
    f.write("DSV41ST3", 8);
    const int32_t hdr[3] = {L, T, H4};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    for (const auto& st : o.layer_states) {
      if (st.size() != static_cast<size_t>(T) * H4) throw std::runtime_error("a layer capture with the wrong rows (bounded?)");
      f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
    }
    for (const auto& p : pre) {
      if (p.size() != static_cast<size_t>(T) * 4) throw std::runtime_error("a pre capture with the wrong rows");
      f.write(reinterpret_cast<const char*>(p.data()), static_cast<std::streamsize>(p.size() * 4));
    }
    f.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 8));
    const int32_t K = static_cast<int32_t>(cfg.num_experts_per_tok);
    f.write(reinterpret_cast<const char*>(&K), 4);
    if (static_cast<int>(o.route_ids.size()) < L) throw std::runtime_error("route captures");
    for (int l = 0; l < L; ++l) {
      if (o.route_ids[static_cast<size_t>(l)].size() != static_cast<size_t>(T) * K) throw std::runtime_error("route rows");
      f.write(reinterpret_cast<const char*>(o.route_ids[static_cast<size_t>(l)].data()), static_cast<std::streamsize>(T) * K * 4);
    }
    const int32_t Li = static_cast<int32_t>(o.dsa_selections.size());
    const int32_t ms = static_cast<int32_t>(cfg.index_topk);
    f.write(reinterpret_cast<const char*>(&Li), 4);
    f.write(reinterpret_cast<const char*>(&ms), 4);
    for (const auto& sel : o.dsa_selections) {
      if (sel.size() != static_cast<size_t>(T) * ms) throw std::runtime_error("selection rows");
      f.write(reinterpret_cast<const char*>(sel.data()), static_cast<std::streamsize>(sel.size() * 4));
    }
    DGPP_LOG_INFO("dsv41_forward_check: {} layers x {} rows x {} (+ routes, {} index sources) written to {} in {:.1f} s", L, T, H4,
                  Li, out, secs);
    if (layers <= 0 && !o.logits.empty()) {
      const int V = o.lm_vocab_count;
      const float* last = o.logits.data() + o.logits.size() - static_cast<size_t>(V);
      std::vector<int> order(static_cast<size_t>(V));
      for (int i = 0; i < V; ++i) order[static_cast<size_t>(i)] = i;
      std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return last[a] > last[b]; });
      std::string s;
      for (int i = 0; i < 8; ++i) s += std::format("{}{}:{:.3f}", i ? " " : "", order[static_cast<size_t>(i)], last[order[static_cast<size_t>(i)]]);
      DGPP_LOG_INFO("dsv41_forward_check: the last row's top-8 logits (id:value): {}", s);
    }
    return 0;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("dsv41_forward_check: {}", e.what());
    return 1;
  }
}
