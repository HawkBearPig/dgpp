// Opt-in research worker: a private owner socket and DGPP journal/collective ranks.
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "adapters/artifact.hpp"
#include "adapters/runtime.hpp"
#include "common/cuda_check.hpp"
#include "common/process_memory.hpp"
#include "heretic_metadata.hpp"
#include "heretic_world.hpp"
#ifdef DGPP_HERETIC_FABRIC
#include "adapters/bus_sum.hpp"
#include "dgpp_version.hpp"
#include "engine/graph_engine.hpp"
#endif
#include "loaders/minijson.hpp"
#include "models/glm_dsa/model.hpp"

namespace {
using dgpp::minijson::Value;
constexpr size_t kHeaderLimit = 64 * 1024;
constexpr size_t kPayloadLimit = 64 * 1024 * 1024;
constexpr int kPromptLimit = 2048;
constexpr size_t kMemoryReserve = size_t{8} << 30;
static_assert(std::endian::native == std::endian::little);

void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
void transfer(int fd, void* data, size_t bytes, bool sending) {
  auto* p = static_cast<char*>(data);
  while (bytes) {
    const auto n = sending ? send(fd, p, bytes, MSG_NOSIGNAL) : recv(fd, p, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    require(n > 0, sending ? "control write failed" : "control channel closed or truncated");
    p += n;
    bytes -= n;
  }
}
int64_t integer(const Value& v) {
  require(v.kind() == Value::Kind::Int, "expected an integer");
  return v.as_int();
}
std::string string(const Value& v) {
  require(v.is_string(), "expected a string");
  return std::string(v.as_string());
}
void fields(const Value& object, const std::set<std::string>& allowed) {
  require(object.is_object(), "expected an object");
  std::set<std::string> seen;
  for (const auto& m : object.members())
    require(allowed.count(m.key) && seen.insert(m.key).second, "unknown or duplicate field");
}
std::string quoted(const std::string& text) {
  std::string out;
  dgpp::serve::append_json_string(&out, text);
  return out;
}
struct Reply {
  std::string extra;
  std::string tensors;
  std::vector<uint8_t> payload;
  template <class T>
  void tensor(const std::string& name, const char* dtype, const std::vector<T>& data,
              std::initializer_list<int64_t> shape) {
    const size_t bytes = data.size() * sizeof(T);
    require(bytes <= kPayloadLimit - payload.size(), "response exceeds tensor budget");
    if (!tensors.empty()) tensors += ',';
    tensors += "{\"name\":" + quoted(name) + ",\"dtype\":" + quoted(dtype) + ",\"shape\":[";
    bool first = true;
    for (auto dim : shape) {
      if (!first) tensors += ',';
      first = false;
      tensors += std::to_string(dim);
    }
    tensors += "],\"offset\":" + std::to_string(payload.size()) +
               ",\"nbytes\":" + std::to_string(bytes) + "}";
    const auto* begin = reinterpret_cast<const uint8_t*>(data.data());
    payload.insert(payload.end(), begin, begin + bytes);
  }
  void send_to(int fd, int64_t seq, const std::string& op, int64_t epoch,
               const std::string& candidate_id) {
    auto header = "{\"schema\":\"dgpp.heretic.rpc\",\"version\":2,\"seq\":" + std::to_string(seq) +
                  ",\"op\":" + quoted(op) + ",\"epoch\":" + std::to_string(epoch) +
                  ",\"candidate_id\":" + quoted(candidate_id) +
                  ",\"state\":" + quoted(candidate_id.empty() ? "baseline" : "candidate") +
                  ",\"tensors\":[" + tensors + "]" + extra + "}";
    require(header.size() <= kHeaderLimit, "response header exceeds budget");
    std::array<uint32_t, 2> lengths{static_cast<uint32_t>(header.size()),
                                    static_cast<uint32_t>(payload.size())};
    transfer(fd, lengths.data(), sizeof(lengths), true);
    transfer(fd, header.data(), header.size(), true);
    transfer(fd, payload.data(), payload.size(), true);
  }
};
Reply assemble_prediction(const std::vector<std::string>& pieces, const dgpp::GlmDsaTextConfig& cfg,
                          int world, bool capture, bool adapted) {
  require(pieces.size() == static_cast<size_t>(world), "incomplete research world reply");
  const auto targets = dgpp::glm_dsa_modification_targets(cfg, 0, world);
  std::vector<float> logits;
  std::vector<uint16_t> residuals;
  std::vector<std::vector<uint16_t>> inputs(targets.size());
  std::vector<std::vector<float>> outputs(targets.size());
  for (int rank = 0; rank < world; ++rank) {
    dgpp::heretic::TensorReader reader(pieces[rank]);
    const auto sharding =
        world == 1 ? dgpp::GlmDsaHeadSharding::Full : dgpp::GlmDsaHeadSharding::VocabSharded;
    const auto begin =
        dgpp::GlmDsaLocalGeometry::from_config(cfg, rank, world, sharding).lm_vocab_begin;
    const auto count = dgpp::GlmDsaLayerStream::lm_vocab_count(cfg, rank, world, sharding);
    require(begin == static_cast<int64_t>(logits.size()), "non-contiguous vocabulary partition");
    const auto shard = reader.take<float>(count);
    logits.insert(logits.end(), shard.begin(), shard.end());
    if (capture) {
      auto hidden =
          reader.take<uint16_t>(static_cast<size_t>(cfg.num_hidden_layers + 1) * cfg.hidden_size);
      if (rank == 0)
        residuals = std::move(hidden);
      else
        require(hidden == residuals, "residuals disagree across ranks");
      for (size_t i = 0; i < targets.size(); ++i) {
        const auto& t = targets[i];
        auto x = reader.take<uint16_t>(t.input_features / world);
        inputs[i].insert(inputs[i].end(), x.begin(), x.end());
        auto y = reader.take<float>(t.output_features);
        if (rank == 0)
          outputs[i] = std::move(y);
        else if (!adapted && t.module.find(".shared_experts.") != std::string::npos) {
          // Canonical FP32 rank-order sum of actual shared down partials.
          for (size_t j = 0; j < y.size(); ++j) outputs[i][j] += y[j];
        } else
          require(y == outputs[i], "replicated projection outputs disagree across ranks");
      }
    }
    reader.finish();
  }
  require(logits.size() == static_cast<size_t>(cfg.vocab_size),
          "incomplete full-vocabulary logits");
  Reply reply;
  reply.tensor("logits", "float32", logits, {cfg.vocab_size});
  if (capture) {
    reply.tensor("residuals", "bfloat16", residuals, {cfg.num_hidden_layers + 1, cfg.hidden_size});
    for (size_t i = 0; i < targets.size(); ++i) {
      reply.tensor(targets[i].id + ".input", "bfloat16", inputs[i], {targets[i].input_features});
      reply.tensor(targets[i].id + ".output", "float32", outputs[i], {targets[i].output_features});
    }
  }
  return reply;
}

void validate_config(const std::string& path, const dgpp::serve::ClusterConfig& c) {
  require(c.engine.heretic_research,
          "Heretic research is disabled; set engine.heretic_research=true");
  require(c.nodes.size() == 1 || c.nodes.size() == 2 || c.nodes.size() == 4,
          "research worker requires world size 1, 2 or 4");
#ifndef DGPP_HERETIC_FABRIC
  require(c.nodes.size() == 1, "distributed research requires an IBV-enabled build");
#endif
  require(c.engine.max_concurrency == 1 && !c.engine.mtp && !c.engine.decode_graph &&
              c.engine.prefix_cache_gib == 0,
          "research worker requires max_concurrency=1, mtp=false, decode_graph=false, "
          "prefix_cache_gib=0");
  require(c.engine.kv_dtype == "bf16" && c.engine.bf16_weights == "checkpoint" &&
              c.engine.attention_weights == "checkpoint",
          "research worker requires BF16 KV and checkpoint weight formats");
  // Reject explicit serving options this separate worker does not implement.
  // In particular, never silently disregard a requested numerical recipe.
  std::ifstream f(path);
  const std::string bytes{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
  const auto parsed = dgpp::minijson::parse_document(bytes);
  fields(parsed.root.at("engine"),
         {"heretic_research", "runtime_adapters", "max_concurrency", "mtp", "decode_graph",
          "prefix_cache_gib", "kv_capacity", "kv_dtype", "bf16_weights", "attention_weights",
          "default_max_tokens"});
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string config_path;
    int fd = -1, rank = 0;
    std::string op_log;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::puts(
            "dgpp-heretic-worker --config FILE [--rank N] [--control-fd FD] [--op-log FILE]\n"
            "Rank 0 owns the inherited Unix socket; peers use the configured journal/fabric "
            "ports.");
        return 0;
      }
      require(
          (arg == "--config" || arg == "--control-fd" || arg == "--rank" || arg == "--op-log") &&
              i + 1 < argc,
          "expected --config FILE, --rank N, --control-fd FD, or --op-log FILE");
      const std::string value = argv[++i];
      if (arg == "--config")
        config_path = value;
      else if (arg == "--op-log")
        op_log = value;
      else {
        int& number = arg == "--rank" ? rank : fd;
        const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), number);
        require(ec == std::errc{} && end == value.data() + value.size(), "invalid rank/control fd");
      }
    }
    require(!config_path.empty(), "--config is required");
    const auto cluster = dgpp::serve::load_cluster_config(config_path);
    validate_config(config_path, cluster);  // before checkpoint access or CUDA initialization
    const int world = static_cast<int>(cluster.nodes.size());
    require(rank >= 0 && rank < world, "rank is outside this research world");
    int socket_type = 0;
    socklen_t socket_type_len = sizeof(socket_type);
    sockaddr_storage address{};
    socklen_t address_len = sizeof(address);
    require(
        rank != 0 ||
            (fd >= 3 && getsockopt(fd, SOL_SOCKET, SO_TYPE, &socket_type, &socket_type_len) == 0 &&
             socket_type == SOCK_STREAM &&
             getsockname(fd, reinterpret_cast<sockaddr*>(&address), &address_len) == 0 &&
             address.ss_family == AF_UNIX),
        "--control-fd must be an inherited Unix stream socket");
    const auto checkpoint = dgpp::heretic::checkpoint_path(cluster, rank);
    const auto cfg = dgpp::GlmDsaTextConfig::from_json_file(checkpoint + "/config.json");
    const auto catalog = dgpp::heretic::catalog_json(cluster, checkpoint, cfg, 0);
    const int context = static_cast<int>(
        std::min<int64_t>(cluster.engine.kv_capacity, cfg.max_position_embeddings));
    const int max_prompt = std::min(context, kPromptLimit);
    require(max_prompt > 0, "context capacity must be positive");
    std::string identity_material =
        dgpp::heretic::file_bytes(config_path) +
        dgpp::heretic::file_bytes(checkpoint + "/config.json") +
        dgpp::heretic::file_bytes(checkpoint + "/model.safetensors.index.json", true);
    const auto raw_config =
        dgpp::minijson::parse_document(dgpp::heretic::file_bytes(checkpoint + "/config.json"));
    if (const auto* quant = raw_config.root.find("quantization_config");
        quant && quant->is_object()) {
      const auto* method = quant->find("quant_method");
      if (method && string(*method) == "dgpp_mixed346") {
        for (const auto& [block, key, fallback] :
             {std::tuple{"baseline", "baseline_config_file", "baseline-quantization-config.json"},
              std::tuple{"recipe", "recipe_file", "quantization-recipe.json"}}) {
          if (const auto* inline_value = quant->find(block);
              inline_value && !inline_value->is_null())
            continue;
          const auto* name = quant->find(key);
          identity_material +=
              dgpp::heretic::file_bytes(checkpoint + "/" + (name ? string(*name) : fallback));
        }
      }
    }
    const auto revision = std::filesystem::path(checkpoint).filename().string();
    if (revision.size() == 40 &&
        revision.find_first_not_of("0123456789abcdef") == std::string::npos)
      identity_material += revision;
    identity_material += dgpp::heretic::binary_digest();
    identity_material += dgpp::adapters::numerical_environment();
#ifdef DGPP_HERETIC_FABRIC
    identity_material += DGPP_VERSION;
    identity_material += DGPP_GIT_SHA;
    std::unique_ptr<dgpp::net::CollectiveBus> bus;
    std::unique_ptr<dgpp::BusBoundaryReducer> reducer;
    struct PickScratch {
      uint16_t* data = nullptr;
      ~PickScratch() {
        if (data) cudaFreeHost(data);
      }
    } pick_scratch;
#endif
    dgpp::heretic::ResearchWorld ranks(rank, world, fd, cluster,
                                       dgpp::serve::config_digest(identity_material), op_log);
    std::unique_ptr<dgpp::GlmDsaModel> model;
    std::optional<dgpp::adapters::CheckpointIdentity> base_identity;
    std::optional<dgpp::adapters::Manifest> staged;
    int64_t epoch = 0;
    std::string candidate_id;
    auto identity = [&]() -> const dgpp::adapters::CheckpointIdentity& {
      require(cluster.engine.runtime_adapters,
              "checkpoint identity requires engine.runtime_adapters=true");
      if (!base_identity) {
        base_identity = dgpp::adapters::fingerprint_checkpoint(checkpoint);
        const auto peers = ranks.gather(base_identity->sha256);
        for (const auto& peer : peers)
          require(peer == base_identity->sha256, "checkpoint bytes disagree across ranks");
        ranks.barrier();
      }
      base_identity->check_unchanged(checkpoint);
      return *base_identity;
    };
    auto load = [&]() -> dgpp::GlmDsaModel& {
      if (!model) {
        if (cluster.engine.runtime_adapters) (void)identity();
        const auto plan = dgpp::GlmDsaModel::plan_memory(cfg, max_prompt, context, rank, world,
                                                         dgpp::GlmDsaResidency::Resident, 1, false,
                                                         1, dgpp::LatentFormat::kBf16, true);
        const size_t available = dgpp::host_memory_available_bytes();
        require(available > kMemoryReserve && plan.total_bytes() <= available - kMemoryReserve,
                "resident model exceeds available memory with 8 GiB research reserve; distributed "
                "worker required");
        size_t free = 0, total = 0;
        DGPP_CUDA_OK(cudaMemGetInfo(&free, &total));
        int device = 0, integrated = 0;
        DGPP_CUDA_OK(cudaGetDevice(&device));
        DGPP_CUDA_OK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));
        // GB10's host file cache is reclaimable CUDA memory. The serving
        // loader uses MemAvailable for that unified pool as well.
        const size_t device_budget = integrated ? std::max(free, available) : free;
        require(
            device_budget > kMemoryReserve && plan.device_bytes() <= device_budget - kMemoryReserve,
            "resident model exceeds available device memory with 8 GiB research reserve");
        DGPP_LOG_INFO(
            "heretic rank {} memory plan {} bytes; host available {}; device free {}; integrated "
            "{}",
            rank, plan.total_bytes(), available, free, integrated);
        dgpp::BoundaryReducer* boundary = nullptr;
#ifdef DGPP_HERETIC_FABRIC
        if (world > 1) {
          const size_t slot_bytes =
              std::max<size_t>(262144, cfg.hidden_size * 2 * dgpp::kDecodeRows);
          bus = std::make_unique<dgpp::net::CollectiveBus>(dgpp::fabric_bus_options(
              rank, world, cluster.fabric_port, cluster.nodes[0], 120000, slot_bytes));
          std::string error;
          if (!bus->start(&error)) throw std::runtime_error("research fabric startup: " + error);
          reducer =
              std::make_unique<dgpp::BusBoundaryReducer>(*bus, dgpp::heretic::kWorldDeadlineMs);
          boundary = reducer.get();
          DGPP_CUDA_OK(cudaHostAlloc(&pick_scratch.data,
                                     dgpp::kPickScratchElems(world) * sizeof(uint16_t),
                                     cudaHostAllocDefault));
        }
#endif
        model = std::make_unique<dgpp::GlmDsaModel>(
            cfg, checkpoint, max_prompt, context, dgpp::GlmDsaResidency::Resident, boundary, rank,
            world, 1, false, 1, dgpp::LatentFormat::kBf16, true);
        if (base_identity) base_identity->check_unchanged(checkpoint);
        ranks.barrier();  // no rank enters its first collective until every model is loaded
        dgpp::log_memory_ledger("heretic rank " + std::to_string(rank) + " ready");
      }
      return *model;
    };
    try {
      for (int64_t sequence = 1;; ++sequence) {
        std::string header, payload;
        if (ranks.head()) {
          std::array<uint32_t, 2> lengths{};
          char peek;
          ssize_t n;
          do {
            n = recv(fd, &peek, 1, MSG_PEEK);
          } while (n < 0 && errno == EINTR);
          if (n == 0) return 0;
          transfer(fd, lengths.data(), sizeof(lengths), false);
          require(lengths[0] > 0 && lengths[0] <= kHeaderLimit && lengths[1] <= kPayloadLimit,
                  "invalid request frame lengths");
          header.resize(lengths[0]);
          transfer(fd, header.data(), header.size(), false);
          payload.resize(lengths[1]);
          transfer(fd, payload.data(), payload.size(), false);
        }
        if (!ranks.command(&header, &payload)) return 0;
        ranks.active(true);
        const auto parsed = dgpp::minijson::parse_document(header);
        const auto& request = parsed.root;
        const auto op = string(request.at("op"));
        auto allowed = std::set<std::string>{"schema", "version", "seq", "op"};
        if (op == "first_token_logits" || op == "capture" || op == "generate" ||
            op == "score_tokens")
          allowed.insert("token_ids");
        if (op == "score_tokens") allowed.insert("continuation_ids");
        if (op == "generate") {
          allowed.insert("max_new_tokens");
          allowed.insert("stop_token_ids");
        }
        if (op == "read_projection_rows") {
          allowed.insert("target_id");
          allowed.insert("row_begin");
          allowed.insert("row_count");
        }
        if (op == "stage_projection") allowed.insert("index");
        fields(request, allowed);
        require(payload.empty() || op == "begin_candidate" || op == "stage_projection",
                "unexpected request payload");
        require(!staged || op == "stage_projection" || op == "activate_candidate" ||
                    op == "abort_candidate" || op == "reset" || op == "close",
                "candidate staging must finish before another operation");
        require(string(request.at("schema")) == "dgpp.heretic.rpc" &&
                    integer(request.at("version")) == 2 && integer(request.at("seq")) == sequence,
                "protocol version/schema/sequence mismatch");
        Reply reply;
        if (op == "describe") {
          (void)ranks.gather({});
          reply.extra =
              ",\"catalog\":" + catalog +
              ",\"capabilities\":[\"describe_targets\",\"first_token_logits\","
              "\"capture_last_token\",\"generate_greedy\",\"read_projection_rows\",\"score_"
              "tokens\"" +
              std::string(cluster.engine.runtime_adapters ? ",\"runtime_adapters\"" : "") +
              "],\"vocab_size\":" + std::to_string(cfg.vocab_size) +
              ",\"hidden_size\":" + std::to_string(cfg.hidden_size) +
              ",\"context_limit\":" + std::to_string(context) +
              ",\"max_prompt_tokens\":" + std::to_string(max_prompt) +
              ",\"max_new_tokens\":" + std::to_string(cluster.engine.default_max_tokens) +
              ",\"environment_sha256\":" +
              quoted(dgpp::adapters::sha256(dgpp::adapters::numerical_environment())) +
              ",\"worker_sha256\":" + quoted(dgpp::adapters::file_sha256("/proc/self/exe"));
        } else if (op == "checkpoint_identity") {
          reply.extra = ",\"base\":" + identity().json();
          ranks.barrier();
        } else if (op == "begin_candidate") {
          staged = dgpp::adapters::parse_manifest(payload, cfg, identity(), world);
          ranks.barrier();
        } else if (op == "stage_projection") {
          require(cluster.engine.runtime_adapters && staged.has_value(), "no candidate is staged");
          const auto index = integer(request.at("index"));
          require(index >= 0 && static_cast<size_t>(index) < staged->projections.size(),
                  "invalid projection index");
          auto& p = staged->projections[index];
          const size_t a_bytes = p.rank * p.target.input_features * sizeof(float);
          const size_t b_bytes = p.rank * p.target.output_features * sizeof(float);
          require(payload.size() == a_bytes + b_bytes, "projection payload size mismatch");
          dgpp::adapters::set_factors(p, payload.data(), a_bytes, payload.data() + a_bytes,
                                      b_bytes);
          ranks.barrier();
        } else if (op == "activate_candidate") {
          require(cluster.engine.runtime_adapters && staged.has_value(), "no candidate is staged");
          auto& m = load();
          (void)identity();
          const size_t bytes = dgpp::adapters::Runtime::factor_bytes(*staged, world);
          const size_t available = dgpp::host_memory_available_bytes();
          require(available > kMemoryReserve && bytes < available - kMemoryReserve,
                  "runtime adapter exceeds available memory reserve");
          dgpp::adapters::SumFp32 sum;
#ifdef DGPP_HERETIC_FABRIC
          if (bus)
            sum = dgpp::adapters::bus_sum(
                *bus, rank, world,
                std::max<size_t>(max_prompt * dgpp::adapters::kMaxAdapterRank, cfg.hidden_size),
                dgpp::heretic::kWorldDeadlineMs);
#endif
          auto adapter = std::make_unique<dgpp::adapters::Runtime>(
              *staged, rank, world, max_prompt, cfg.hidden_size, cfg.num_hidden_layers,
              std::move(sum), cluster.engine.runtime_adapters);
          ranks.barrier();
          m.set_runtime_adapter(std::move(adapter), cluster.engine.runtime_adapters);
          candidate_id = staged->id;
          staged.reset();
          ++epoch;
          ranks.barrier();
        } else if (op == "abort_candidate" || op == "reset") {
          require(cluster.engine.runtime_adapters, "runtime adapters are disabled");
          staged.reset();
          if (op == "reset") {
            if (model) model->set_runtime_adapter(nullptr, cluster.engine.runtime_adapters);
            candidate_id.clear();
            ++epoch;
          }
          ranks.barrier();
        } else if (op == "read_projection_rows") {
          const auto id = string(request.at("target_id"));
          const auto targets = dgpp::glm_dsa_modification_targets(cfg, 0, world);
          const auto t = std::find_if(targets.begin(), targets.end(),
                                      [&](const auto& t) { return t.id == id; });
          require(t != targets.end(), "unknown projection target");
          const auto begin = integer(request.at("row_begin")),
                     rows = integer(request.at("row_count"));
          require(begin >= 0 && rows > 0 && rows <= t->output_features &&
                      begin <= t->output_features - rows,
                  "projection row slice outside target");
          require(rows <= (32 * 1024 * 1024) / (4 * t->input_features),
                  "projection slice exceeds 32 MiB budget");
          std::string wire;
          dgpp::heretic::append_values(wire, load().research_projection_rows(
                                                 id, begin, rows, cluster.engine.heretic_research));
          const auto pieces = ranks.gather(wire);
          if (ranks.head()) {
            const size_t cols = t->input_features / world;
            std::vector<float> weights(rows * t->input_features);
            for (int peer = 0; peer < world; ++peer) {
              dgpp::heretic::TensorReader reader(pieces[peer]);
              const auto local = reader.take<float>(rows * cols);
              reader.finish();
              for (int64_t r = 0; r < rows; ++r)
                std::copy_n(local.data() + r * cols, cols,
                            weights.data() + r * t->input_features + peer * cols);
            }
            reply.tensor("weight", "float32", weights, {rows, t->input_features});
          }
        } else if (op == "close") {
          ranks.active(false);
          (void)ranks.gather({});
          if (ranks.head()) reply.send_to(fd, sequence, op, epoch, candidate_id);
          return 0;
        } else {
          require(op == "first_token_logits" || op == "capture" || op == "generate" ||
                      op == "score_tokens",
                  "unsupported operation");
          const auto& tokens = request.at("token_ids");
          require(tokens.is_array() && !tokens.items().empty() &&
                      tokens.items().size() <= static_cast<size_t>(max_prompt),
                  "prompt token count exceeds limits or is empty");
          std::vector<int64_t> ids;
          for (const auto& value : tokens.items()) {
            const auto id = integer(value);
            require(id >= 0 && id < cfg.vocab_size, "invalid token ID");
            ids.push_back(id);
          }
          if (op == "score_tokens") {
            const auto& continuation = request.at("continuation_ids");
            require(continuation.is_array() && !continuation.items().empty() &&
                        continuation.items().size() <= 2048 &&
                        continuation.items().size() <= static_cast<size_t>(context) - ids.size(),
                    "continuation exceeds score token budget");
            std::vector<int64_t> expected;
            for (const auto& value : continuation.items()) {
              const auto token = integer(value);
              require(token >= 0 && token < cfg.vocab_size, "invalid continuation token ID");
              expected.push_back(token);
            }
            auto& m = load();
            auto out = m.session_prefill(0, ids);
            std::vector<float> logprobs;
            std::vector<int64_t> greedy;
            for (size_t i = 0; i < expected.size(); ++i) {
              std::string wire;
              dgpp::heretic::append_values(wire, out.logits);
              const auto pieces = ranks.gather(wire);
              if (ranks.head()) {
                std::vector<float> logits;
                for (int peer = 0; peer < world; ++peer) {
                  const auto sharding = world == 1 ? dgpp::GlmDsaHeadSharding::Full
                                                   : dgpp::GlmDsaHeadSharding::VocabSharded;
                  dgpp::heretic::TensorReader tensor(pieces[peer]);
                  const auto shard = tensor.take<float>(
                      dgpp::GlmDsaLayerStream::lm_vocab_count(cfg, peer, world, sharding));
                  tensor.finish();
                  logits.insert(logits.end(), shard.begin(), shard.end());
                }
                require(logits.size() == static_cast<size_t>(cfg.vocab_size),
                        "score logits shape mismatch");
                for (float v : logits) require(std::isfinite(v), "non-finite score logits");
                const auto best = std::max_element(logits.begin(), logits.end());
                double sum = 0;
                for (float value : logits) sum += std::exp(static_cast<double>(value) - *best);
                logprobs.push_back(static_cast<double>(logits[expected[i]]) - *best -
                                   std::log(sum));
                greedy.push_back(best - logits.begin() == expected[i]);
              }
              ranks.token(expected[i]);
              if (i + 1 < expected.size()) out = m.session_step(0, expected[i]);
            }
            m.session_close(0);
            ranks.barrier();
            if (ranks.head()) {
              reply.tensor("logprobs", "float32", logprobs,
                           {static_cast<int64_t>(expected.size())});
              reply.tensor("greedy", "int64", greedy, {static_cast<int64_t>(expected.size())});
            }
          } else if (op == "generate") {
            const auto count = integer(request.at("max_new_tokens"));
            require(count > 0 && count <= cluster.engine.default_max_tokens &&
                        count <= context - static_cast<int64_t>(ids.size()),
                    "generation exceeds configured token budget");
            const auto& stops = request.at("stop_token_ids");
            require(stops.is_array() && stops.items().size() <= 64, "invalid stop token list");
            std::set<int64_t> stop_ids;
            for (const auto& value : stops.items()) {
              const auto id = integer(value);
              require(id >= 0 && id < cfg.vocab_size, "invalid stop token ID");
              stop_ids.insert(id);
            }
            auto& m = load();
            auto out = m.session_prefill(0, ids);
            std::vector<int64_t> generated;
            bool stopped = false;
            for (int64_t i = 0; i < count; ++i) {
              require(out.logits.size() == static_cast<size_t>(out.lm_vocab_count),
                      "generation logits shape mismatch");
              for (float v : out.logits) require(std::isfinite(v), "non-finite generation logits");
              int64_t token = std::max_element(out.logits.begin(), out.logits.end()) -
                              out.logits.begin() + out.lm_vocab_begin;
#ifdef DGPP_HERETIC_FABRIC
              if (world > 1)
                token = dgpp::bus_greedy_pick(
                    *bus, rank, world,
                    dgpp::sample::local_max(out.logits.data(), out.lm_vocab_count,
                                            out.lm_vocab_begin),
                    pick_scratch.data, dgpp::heretic::kWorldDeadlineMs);
#endif
              ranks.token(token);
              generated.push_back(token);
              if (stop_ids.count(token)) {
                stopped = true;
                break;
              }
              if (i + 1 < count) out = m.session_step(0, token);
            }
            m.session_close(0);
            std::string wire;
            dgpp::heretic::append_values(wire, generated);
            for (const auto& peer : ranks.gather(wire))
              require(peer == wire, "generated tokens disagree across ranks");
            reply.tensor("token_ids", "int64", generated, {static_cast<int64_t>(generated.size())});
            reply.extra = ",\"stop_reason\":" + quoted(stopped ? "stop_token" : "length");
          } else {
            std::string wire;
            auto& m = load();
            if (op == "capture") {
              const auto out = m.research_forward(ids, cluster.engine.heretic_research);
              dgpp::heretic::append_values(wire, out.prediction.logits);
              dgpp::heretic::append_values(wire, out.residuals);
              for (const auto& p : out.projections) {
                dgpp::heretic::append_values(wire, p.input);
                dgpp::heretic::append_values(wire, p.output);
              }
            } else {
              const auto out = m.session_prefill(0, ids);
              m.session_close(0);
              dgpp::heretic::append_values(wire, out.logits);
            }
            const auto pieces = ranks.gather(wire);
            if (ranks.head())
              reply =
                  assemble_prediction(pieces, cfg, world, op == "capture", m.has_runtime_adapter());
          }
        }
        ranks.active(false);
        if (ranks.head()) reply.send_to(fd, sequence, op, epoch, candidate_id);
      }
    } catch (const std::exception& e) {
      // Never unwind through a CUDA destructor after a rank/collective failure:
      // the peer may be unable to complete the stream we would synchronize.
      std::fprintf(stderr, "dgpp-heretic-worker rank %d: %s\n", rank, e.what());
      std::fflush(stderr);
      std::_Exit(1);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dgpp-heretic-worker: %s\n", e.what());
    return 1;  // Fail closed: no subsequent request can observe partial model state.
  }
}
