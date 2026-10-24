// flint_load_check: loads a Flint checkpoint through the resident
// loader at one rank's TP geometry and reports per-layer bytes and times,
// the byte formulas against actual usage, the source-byte plan against the
// bytes read, the bind report's ignored count (the vision / audio encoders
// and the later draft layers), and the replicated digest (compare across
// ranks by hand or by the fabric script). No bus, no forward: one process
// per rank.
//
//   flint_load_check --model ORG/NAME | --checkpoint-dir DIR
//                   [--world W] [--rank R] [--streaming]
//                   [--layers N] [--image-dir DIR|off]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <unordered_map>

#include "common/log.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/safetensors.hpp"
#include "models/flint/binding.hpp"
#include "models/flint/config.hpp"
#include "models/flint/loader.hpp"

int main(int argc, char** argv) {
  std::string model_id, ckpt, image_dir;
  int world = 1, rank = 0, layers = -1;
  bool streaming = false;
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
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--image-dir") image_dir = next(i);
      else throw std::runtime_error("unknown argument " + a);
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    if (dgpp::detect_architecture_file(cfg_path) != dgpp::ModelArchitecture::Flint)
      throw std::runtime_error("not a Flint checkpoint: " + ckpt);
    const dgpp::FlintTextConfig cfg = dgpp::FlintTextConfig::from_json_file(cfg_path);
    if (!image_dir.empty()) dgpp::FlintLayerStream::set_resident_image_dir(image_dir == "off" ? "" : image_dir);
    const dgpp::FlintResidency residency = streaming ? dgpp::FlintResidency::Streaming : dgpp::FlintResidency::Resident;
    const dgpp::FlintHeadSharding head = world > 1 ? dgpp::FlintHeadSharding::VocabSharded : dgpp::FlintHeadSharding::Full;
    const double kGiB = 1024.0 * 1024.0 * 1024.0;
    DGPP_LOG_INFO("flint_load_check: {} world {} rank {} {} — {} layers ({} sliding / {} full, {} MoE), formulas: resident {:.2f} GiB (globals {:.2f})",
                  ckpt, world, rank, streaming ? "streaming" : "resident", cfg.num_hidden_layers, cfg.num_swa_layers(),
                  cfg.num_hidden_layers - cfg.num_swa_layers(), cfg.num_moe_layers(),
                  dgpp::FlintLayerStream::resident_bytes(cfg, rank, world, head, false) / kGiB,
                  dgpp::FlintLayerStream::globals_bytes(cfg, rank, world, head) / kGiB);
    // The bind report over every shard's header, with the ignored count.
    {
      std::unordered_map<std::string, dgpp::FlintTensorDesc> present;
      for (const auto& entry : std::filesystem::directory_iterator(ckpt)) {
        if (entry.path().extension() != ".safetensors") continue;
        auto f = dgpp::SafetensorsFile::open(entry.path().string());
        f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::FlintTensorDesc{t.dtype, t.shape}); });
      }
      const dgpp::FlintBindReport rep = dgpp::flint_validate_text_binding(cfg, present);
      DGPP_LOG_INFO("flint_load_check: binding {} — expected {} matched {} missing {} dtype {} shape {} unexpected {} (fp8 {} fp4 {})",
                    rep.ok() ? "ok" : "FAILED", rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch,
                    rep.unexpected, rep.fp8_matrices, rep.fp4_matrices);
      for (const std::string& e : rep.errors) DGPP_LOG_ERROR("flint_load_check: {}", e);
    }
    const auto t0 = std::chrono::steady_clock::now();
    dgpp::FlintLayerStream stream(cfg, ckpt, rank, world, residency, head);
    const auto t1 = std::chrono::steady_clock::now();
    DGPP_LOG_INFO("flint_load_check: opened in {:.1f} s; layer capacity {:.2f} GiB",
                  std::chrono::duration<double>(t1 - t0).count(), stream.layer_capacity() / kGiB);
    const dgpp::FlintReplicatedDigest d = stream.hash_replicated();
    DGPP_LOG_INFO("flint_load_check: digest globals {:016x} tensors {} bytes {:.2f} GiB; layer 0 {:016x} layer 1 {:016x} last {:016x}",
                  d.globals, d.tensors, d.bytes / kGiB, d.layer[0], d.layer[1], d.layer.back());
    const auto& g = stream.load_globals();
    DGPP_LOG_INFO("flint_load_check: globals {:.2f} GiB, lm head rows [{}, +{})", g.bytes / kGiB, g.lm_vocab_begin, g.lm_vocab_count);
    const int n = layers < 0 ? cfg.num_hidden_layers : layers;
    size_t total = 0;
    for (int l = 0; l < n; ++l) {
      const auto tl = std::chrono::steady_clock::now();
      const uint64_t before = stream.source_bytes_read();
      const auto& r = stream.load_layer(l);
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count();
      total += r.bytes;
      if (l < 4 || l == n - 1 || l % 12 == 0)
        DGPP_LOG_INFO("flint_load_check: layer {} ({} {}{}) {:.3f} GiB in {:.2f} s, read {:.3f} GiB; heads {} x kv {} (fused qkv {} rows); {}",
                      l, r.swa ? "sliding" : "full", r.moe ? "moe" : "dense", r.rope ? "" : " no-rope", r.bytes / kGiB, s,
                      (stream.source_bytes_read() - before) / kGiB, r.attn.local_heads, r.attn.local_kv_heads,
                      r.attn.qkv_rows(),
                      r.moe ? std::format("experts {} x inter {} (fp4 mask {}), shared inter {}", r.moe_w.experts_fp8.size() / 3,
                                          r.moe_w.local_inter, r.moe_w.fp4_mask, r.moe_w.shared.local_inter)
                            : std::format("dense inter {}", r.dense.local_inter));
      if (!streaming) continue;
      stream.release_layer();
    }
    if (!streaming) stream.release_sources();
    DGPP_LOG_INFO("flint_load_check: {} layers {:.2f} GiB resident (+ globals {:.2f} = {:.2f} GiB); source bytes read {:.2f} GiB (verbatim {:.2f}); image restored {} captured {}; total {:.1f} s",
                  n, total / kGiB, g.bytes / kGiB, (total + g.bytes) / kGiB,
                  stream.source_bytes_read() / kGiB, stream.verbatim_source_bytes() / kGiB,
                  stream.image_layers_restored(), stream.image_layers_captured(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("flint_load_check: {}", e.what());
    return 1;
  }
}
