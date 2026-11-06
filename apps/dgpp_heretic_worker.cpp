// Private, single-process research channel. No serving listener or engine hooks.
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/process_memory.hpp"
#include "heretic_metadata.hpp"
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
  void send_to(int fd, int64_t seq, const std::string& op) {
    auto header = "{\"schema\":\"dgpp.heretic.rpc\",\"version\":1,\"seq\":" + std::to_string(seq) +
                  ",\"op\":" + quoted(op) + ",\"state\":\"baseline\",\"tensors\":[" + tensors +
                  "]" + extra + "}";
    require(header.size() <= kHeaderLimit, "response header exceeds budget");
    std::array<uint32_t, 2> lengths{static_cast<uint32_t>(header.size()),
                                    static_cast<uint32_t>(payload.size())};
    transfer(fd, lengths.data(), sizeof(lengths), true);
    transfer(fd, header.data(), header.size(), true);
    transfer(fd, payload.data(), payload.size(), true);
  }
};
void validate_config(const std::string& path, const dgpp::serve::ClusterConfig& c) {
  require(c.engine.heretic_research,
          "Heretic research is disabled; set engine.heretic_research=true");
  require(c.nodes.size() == 1,
          "research worker currently supports world_size=1 only; distributed execution is pending");
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
         {"heretic_research", "max_concurrency", "mtp", "decode_graph", "prefix_cache_gib",
          "kv_capacity", "kv_dtype", "bf16_weights", "attention_weights", "default_max_tokens"});
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string config_path;
    int fd = -1;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::puts(
            "dgpp-heretic-worker --config FILE --control-fd FD\n"
            "Opt-in world-1 research worker over an inherited Unix stream socket. No network "
            "listener.");
        return 0;
      }
      require((arg == "--config" || arg == "--control-fd") && i + 1 < argc,
              "expected --config FILE or --control-fd FD");
      const std::string value = argv[++i];
      if (arg == "--config")
        config_path = value;
      else {
        const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), fd);
        require(ec == std::errc{} && end == value.data() + value.size(), "invalid control fd");
      }
    }
    require(!config_path.empty(), "--config is required");
    const auto cluster = dgpp::serve::load_cluster_config(config_path);
    validate_config(config_path, cluster);  // before checkpoint access or CUDA initialization
    int socket_type = 0;
    socklen_t socket_type_len = sizeof(socket_type);
    sockaddr_storage address{};
    socklen_t address_len = sizeof(address);
    require(fd >= 3 && getsockopt(fd, SOL_SOCKET, SO_TYPE, &socket_type, &socket_type_len) == 0 &&
                socket_type == SOCK_STREAM &&
                getsockname(fd, reinterpret_cast<sockaddr*>(&address), &address_len) == 0 &&
                address.ss_family == AF_UNIX,
            "--control-fd must be an inherited Unix stream socket");
    const auto checkpoint = dgpp::heretic::checkpoint_path(cluster, 0);
    const auto cfg = dgpp::GlmDsaTextConfig::from_json_file(checkpoint + "/config.json");
    const auto catalog = dgpp::heretic::catalog_json(cluster, checkpoint, cfg, 0);
    const int context = static_cast<int>(
        std::min<int64_t>(cluster.engine.kv_capacity, cfg.max_position_embeddings));
    const int max_prompt = std::min(context, kPromptLimit);
    require(max_prompt > 0, "context capacity must be positive");
    std::unique_ptr<dgpp::GlmDsaModel> model;
    auto load = [&]() -> dgpp::GlmDsaModel& {
      if (!model) {
        const auto plan = dgpp::GlmDsaModel::plan_memory(cfg, max_prompt, context, 0, 1,
                                                         dgpp::GlmDsaResidency::Resident, 1, false,
                                                         1, dgpp::LatentFormat::kBf16, true);
        const size_t available = dgpp::host_memory_available_bytes();
        require(available > kMemoryReserve && plan.total_bytes() <= available - kMemoryReserve,
                "resident model exceeds available memory with 8 GiB research reserve; distributed "
                "worker required");
        size_t free = 0, total = 0;
        DGPP_CUDA_OK(cudaMemGetInfo(&free, &total));
        require(free > kMemoryReserve && plan.device_bytes() <= free - kMemoryReserve,
                "resident model exceeds available device memory with 8 GiB research reserve");
        model = std::make_unique<dgpp::GlmDsaModel>(cfg, checkpoint, max_prompt, context,
                                                    dgpp::GlmDsaResidency::Resident, nullptr, 0, 1,
                                                    1, false, 1, dgpp::LatentFormat::kBf16, true);
      }
      return *model;
    };
    for (int64_t sequence = 1;; ++sequence) {
      std::array<uint32_t, 2> lengths{};
      // Clean owner disconnect is a normal shutdown; partial frames are fatal.
      char peek;
      ssize_t n;
      do {
        n = recv(fd, &peek, 1, MSG_PEEK);
      } while (n < 0 && errno == EINTR);
      if (n == 0) return 0;
      transfer(fd, lengths.data(), sizeof(lengths), false);
      require(lengths[0] > 0 && lengths[0] <= kHeaderLimit && lengths[1] == 0,
              "invalid request frame lengths");
      std::string header(lengths[0], '\0');
      transfer(fd, header.data(), header.size(), false);
      const auto parsed = dgpp::minijson::parse_document(header);
      const auto& request = parsed.root;
      const auto op = string(request.at("op"));
      auto allowed = std::set<std::string>{"schema", "version", "seq", "op"};
      if (op == "first_token_logits" || op == "capture" || op == "generate")
        allowed.insert("token_ids");
      if (op == "generate") {
        allowed.insert("max_new_tokens");
        allowed.insert("stop_token_ids");
      }
      fields(request, allowed);
      require(string(request.at("schema")) == "dgpp.heretic.rpc" &&
                  integer(request.at("version")) == 1 && integer(request.at("seq")) == sequence,
              "protocol version/schema/sequence mismatch");
      Reply reply;
      if (op == "describe") {
        reply.extra = ",\"catalog\":" + catalog +
                      ",\"capabilities\":[\"describe_targets\",\"first_token_logits\","
                      "\"capture_last_token\",\"generate_greedy\"],\"vocab_size\":" +
                      std::to_string(cfg.vocab_size) +
                      ",\"hidden_size\":" + std::to_string(cfg.hidden_size) +
                      ",\"context_limit\":" + std::to_string(context) +
                      ",\"max_prompt_tokens\":" + std::to_string(max_prompt) +
                      ",\"max_new_tokens\":" + std::to_string(cluster.engine.default_max_tokens);
      } else if (op == "close") {
        reply.send_to(fd, sequence, op);
        return 0;
      } else {
        require(op == "first_token_logits" || op == "capture" || op == "generate",
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
        if (op == "generate") {
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
            require(out.logits.size() == static_cast<size_t>(cfg.vocab_size),
                    "generation logits shape mismatch");
            for (float v : out.logits) require(std::isfinite(v), "non-finite generation logits");
            const auto token =
                std::max_element(out.logits.begin(), out.logits.end()) - out.logits.begin();
            generated.push_back(token);
            if (stop_ids.count(token)) {
              stopped = true;
              break;
            }
            if (i + 1 < count) out = m.session_step(0, token);
          }
          m.session_close(0);
          reply.tensor("token_ids", "int64", generated, {static_cast<int64_t>(generated.size())});
          reply.extra = ",\"stop_reason\":" + quoted(stopped ? "stop_token" : "length");
        } else if (op == "capture") {
          const auto out = load().research_forward(ids, cluster.engine.heretic_research);
          reply.tensor("logits", "float32", out.prediction.logits, {cfg.vocab_size});
          reply.tensor("residuals", "bfloat16", out.residuals,
                       {cfg.num_hidden_layers + 1, cfg.hidden_size});
          for (const auto& p : out.projections) {
            reply.tensor(p.id + ".input", "bfloat16", p.input,
                         {static_cast<int64_t>(p.input.size())});
            reply.tensor(p.id + ".output", "float32", p.output,
                         {static_cast<int64_t>(p.output.size())});
          }
        } else {
          auto& m = load();
          const auto out = m.session_prefill(0, ids);
          m.session_close(0);
          reply.tensor("logits", "float32", out.logits, {cfg.vocab_size});
        }
      }
      reply.send_to(fd, sequence, op);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dgpp-heretic-worker: %s\n", e.what());
    return 1;  // Fail closed: no subsequent request can observe partial model state.
  }
}
