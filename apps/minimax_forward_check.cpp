// Real-checkpoint diagnostics: independent layer dumps and fixed-token
// scoring through the same session verification used by serving.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

#include "common/dtypes.hpp"
#include "engine/graph_engine.hpp"
#include "engine/tp_bus.hpp"
#include "loaders/hf_cache.hpp"
#include "models/minimax/forward.hpp"
#include "net/collective_bus.hpp"
#include "text/tokenizer.hpp"

namespace {
std::string read(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  return {(std::istreambuf_iterator<char>(f)), {}};
}
template <class T>
void write(const std::filesystem::path& path, const std::vector<T>& data) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
  if (!f) throw std::runtime_error("cannot write " + path.string());
}
uint64_t digest(const std::vector<uint16_t>& data) {
  uint64_t h = 1469598103934665603ull;
  for (auto b : data) {
    h ^= b;
    h *= 1099511628211ull;
  }
  return h;
}
void log_digest(const dgpp::MinimaxModel::Outputs& out, int offset, int rank) {
  uint64_t hash = 1469598103934665603ull;
  for (float v : out.logits) {
    hash ^= std::bit_cast<uint32_t>(v);
    hash *= 1099511628211ull;
  }
  std::printf("[minimax_logits] rank %d offset %d rows %zu digest %016llx\n", rank, offset,
              out.logits.size() / out.lm_vocab_count, static_cast<unsigned long long>(hash));
}
void score(const dgpp::MinimaxModel::Outputs& out, int row, int step, int64_t target, int rank) {
  const float* x = out.logits.data() + static_cast<size_t>(row) * out.lm_vocab_count;
  const auto* best = std::max_element(x, x + out.lm_vocab_count);
  const double mx = *best;
  if (!std::isfinite(mx)) throw std::runtime_error("nonfinite logits");
  double sum = 0;
  for (int v = 0; v < out.lm_vocab_count; ++v) sum += std::exp(double(x[v]) - mx);
  const int64_t local = target - out.lm_vocab_begin;
  const double value = local >= 0 && local < out.lm_vocab_count ? x[local] : NAN;
  std::printf(
      "[prefill_tf] rank %d step %d: target %lld argmax %d lmax %.9g lse %.12g target_logit %.9g\n",
      rank, step, static_cast<long long>(target), int(best - x) + out.lm_vocab_begin, mx,
      mx + std::log(sum), value);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    std::string model, checkpoint, text, teacher, ids_text, dump, peer, images;
    int rank = 0, world = 1, port = 29950, rows = 1;
    bool resident = false;
    const auto next = [&](int& i) {
      if (++i >= argc) throw std::runtime_error("missing option value");
      return std::string(argv[i]);
    };
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model")
        model = next(i);
      else if (a == "--checkpoint-dir")
        checkpoint = next(i);
      else if (a == "--text")
        text = next(i);
      else if (a == "--teacher-file")
        teacher = next(i);
      else if (a == "--ids")
        ids_text = next(i);
      else if (a == "--dump-dir")
        dump = next(i);
      else if (a == "--image-dir")
        images = next(i);
      else if (a == "--peer")
        peer = next(i);
      else if (a == "--world")
        world = std::stoi(next(i));
      else if (a == "--rank")
        rank = std::stoi(next(i));
      else if (a == "--port")
        port = std::stoi(next(i));
      else if (a == "--rows")
        rows = std::stoi(next(i));
      else if (a == "--resident")
        resident = true;
      else
        throw std::runtime_error("unknown option " + a);
    }
    if (checkpoint.empty()) {
      std::string error;
      checkpoint = dgpp::hf::model_dir(model, &error);
      if (checkpoint.empty()) throw std::runtime_error(error);
    }
    if (rows < 1 || rows > dgpp::kSpecRows)
      throw std::runtime_error("--rows outside verification capacity");
    if (!teacher.empty()) {
      if (!text.empty() || !ids_text.empty() || !dump.empty())
        throw std::runtime_error("--teacher-file is exclusive with --text, --ids and --dump-dir");
      text = read(teacher);
    }
    std::vector<int64_t> ids;
    if (!text.empty()) {
      const auto tok = dgpp::text::Tokenizer::load(checkpoint + "/tokenizer.json");
      ids = tok.encode(text);
    } else {
      std::istringstream input(ids_text);
      std::string item;
      while (std::getline(input, item, ',')) ids.push_back(std::stoll(item));
    }
    if (ids.empty() || (!teacher.empty() && ids.size() < 2))
      throw std::runtime_error("empty token input");
    const auto cfg = dgpp::MinimaxConfig::from_json_file(checkpoint + "/config.json");
    cfg.validate_tp(rank, world);
    if (!images.empty())
      dgpp::MinimaxLayerStream::set_resident_image_dir(images == "off" ? "" : images);
    const int forward_rows = teacher.empty() ? static_cast<int>(ids.size()) : std::max(rows, 16);
    std::unique_ptr<dgpp::net::CollectiveBus> bus;
    std::unique_ptr<dgpp::BusBoundaryReducer> boundary;
    if (world > 1) {
      if (rank && peer.empty()) throw std::runtime_error("--peer required on peer ranks");
      bus = std::make_unique<dgpp::net::CollectiveBus>(
          dgpp::fabric_bus_options(rank, world, static_cast<uint16_t>(port), peer, 120000,
                                   static_cast<size_t>(forward_rows) * cfg.hidden_size * 2 + 4096));
      std::string error;
      if (!bus->start(&error)) throw std::runtime_error(error);
      boundary = std::make_unique<dgpp::BusBoundaryReducer>(*bus, 600000);
    }
    dgpp::MinimaxModel engine(
        cfg, checkpoint, forward_rows, ids.size() + 128,
        resident ? dgpp::LoaderResidency::Resident : dgpp::LoaderResidency::Streaming,
        boundary.get(), rank, world, 1, false, std::max(rows, 8));
    if (resident) engine.graph_prepare();
    if (!teacher.empty()) {
      auto out = engine.session_prefill(0, {ids.front()});
      log_digest(out, 0, rank);
      score(out, 0, 0, ids[1], rank);
      for (size_t at = 1; at + 1 < ids.size();) {
        const int n = std::min<size_t>(rows, ids.size() - at - 1);
        out = engine.session_verify(0, {ids.begin() + at, ids.begin() + at + n});
        log_digest(out, at, rank);
        for (int row = 0; row < n; ++row) score(out, row, at + row, ids[at + row + 1], rank);
        engine.session_rollback(0, n, n);
        at += n;
        std::fflush(stdout);
      }
      engine.session_close(0);
      std::printf("[prefill_tf_end] rank %d world %d positions %zu\n", rank, world, ids.size() - 1);
    } else {
      engine.set_decode_row_traces(true);
      if (!dump.empty()) {
        const auto path = std::filesystem::path(dump) / ("rank" + std::to_string(rank));
        std::filesystem::create_directories(path);
        engine.set_observer([&, path](int layer, std::string_view stage, const uint16_t* data,
                                      int rows, cudaStream_t stream) {
          std::vector<uint16_t> host(static_cast<size_t>(rows) * cfg.hidden_size);
          DGPP_CUDA_OK(
              cudaMemcpyAsync(host.data(), data, host.size() * 2, cudaMemcpyDeviceToHost, stream));
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          write(path / (std::string(stage) + "-" + std::to_string(layer) + ".bf16"), host);
        });
      }
      const auto out = engine.forward(ids, true);
      for (size_t l = 0; l < out.layer_states.size(); ++l)
        std::printf("layer %zu hidden %016llx\n", l,
                    static_cast<unsigned long long>(digest(out.layer_states[l])));
      for (size_t t = 0; t + 1 < ids.size(); ++t) score(out, t, t, ids[t + 1], rank);
      if (!dump.empty()) {
        const auto path = std::filesystem::path(dump) / ("rank" + std::to_string(rank));
        std::filesystem::create_directories(path);
        write(path / "logits.f32", out.logits);
        for (size_t l = 0, moe = 0; l < out.layer_states.size(); ++l) {
          write(path / ("layer-" + std::to_string(l) + ".bf16"), out.layer_states[l]);
          if (cfg.is_moe_layer(l))
            write(path / ("routes-" + std::to_string(l) + ".i32"), out.route_ids[moe++]);
        }
        std::ofstream meta(path / "metadata.json");
        meta << "{\"world\":" << world << ",\"rank\":" << rank
             << ",\"vocab_begin\":" << out.lm_vocab_begin
             << ",\"vocab_count\":" << out.lm_vocab_count << ",\"tokens\":[";
        for (size_t t = 0; t < ids.size(); ++t) meta << (t ? "," : "") << ids[t];
        meta << "]}\n";
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "minimax_forward_check: %s\n", error.what());
    return 1;
  }
}
