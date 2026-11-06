#pragma once
// Research-only coordination over DGPP's existing journal transport. Model
// arithmetic still uses CollectiveBus; these payloads are cold-path captures.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <vector>

extern char** environ;

#include "loaders/minijson.hpp"
#include "serve/cluster_config.hpp"
#include "serve/fabric_serve.hpp"
#include "serve/json_out.hpp"

namespace dgpp::heretic {
constexpr int kWorldDeadlineMs = 600000;
constexpr size_t kWorldPayloadLimit = 64 * 1024 * 1024;

inline std::string file_bytes(const std::string& path, bool optional = false) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (optional) return {};
    throw std::runtime_error("cannot read research identity file: " + path);
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

inline std::string binary_digest() {
  std::ifstream file("/proc/self/exe", std::ios::binary);
  if (!file) throw std::runtime_error("cannot fingerprint worker binary");
  uint64_t hash = 0xcbf29ce484222325ull;
  char chunk[65536];
  while (file.read(chunk, sizeof(chunk)) || file.gcount()) {
    for (std::streamsize i = 0; i < file.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(chunk[i]);
      hash *= 0x100000001b3ull;
    }
  }
  if (!file.eof()) throw std::runtime_error("cannot finish worker binary fingerprint");
  return std::to_string(hash);
}

inline std::string numerical_environment() {
  // Runtime overrides must agree too. Host transport/cache/log locations are
  // intentionally rank-local; every other DGPP override joins the identity.
  const std::set<std::string> local = {"DGPP_ROCE_DEVICES",
                                       "DGPP_ROCE_GID_INDICES",
                                       "DGPP_RESIDENT_CACHE_DIR",
                                       "DGPP_RESIDENT_CACHE_VERIFY",
                                       "DGPP_LOG_LEVEL",
                                       "DGPP_MLOCK",
                                       "DGPP_NO_SWAP",
                                       "DGPP_NODES",
                                       "DGPP_SSH_USER",
                                       "DGPP_CLUSTER_CONFIG",
                                       "DGPP_HTTP_PORT",
                                       "DGPP_FABRIC_PORT",
                                       "DGPP_JOURNAL_PORT",
                                       "DGPP_LOG_DIR",
                                       "DGPP_STAGE_DIR",
                                       "DGPP_RELEASE_DIR",
                                       "DGPP_HTTP_BIND",
                                       "DGPP_NODE_OVERRIDES",
                                       "DGPP_BUILD_DIR",
                                       "DGPP_DATA_DIR",
                                       "DGPP_ENV_FILE"};
  std::set<std::string> values;
  for (char** entry = environ; entry && *entry; ++entry) {
    const std::string value(*entry);
    if (value.starts_with("DGPP_") && !local.count(value.substr(0, value.find('='))))
      values.insert(value);
  }
  std::string out;
  for (const auto& value : values) {
    out += value;
    out += '\0';
  }
  return out;
}

class ResearchWorld {
 public:
  ResearchWorld(int rank, int world, int owner_fd, const serve::ClusterConfig& cluster,
                const std::string& identity, const std::string& op_log)
      : rank_(rank), owner_fd_(owner_fd) {
    if (!op_log.empty()) {
      audit_.open(op_log);
      if (!audit_) throw std::runtime_error("cannot open research operation log");
    }
    if (world > 1) {
      if (rank == 0) {
        writer_ = std::make_unique<serve::JournalWriter>(cluster.journal_port);
        writer_->accept_peers(world, 120000);
        writer_->broadcast("{\"research_settings\":1,\"heretic_research\":true,\"digest\":\"" +
                           identity + "\"}");
        std::vector<std::pair<int, std::string>> notes;
        if (!writer_->wait_peers_ready(120000, [] { return false; }, &notes))
          throw std::runtime_error("research world readiness cancelled");
        for (const auto& [peer, digest] : notes)
          if (digest != identity)
            throw std::runtime_error("research settings/checkpoint/build mismatch on rank " +
                                     std::to_string(peer));
        writer_->broadcast("{\"research_ready\":1}");
      } else {
        reader_ = std::make_unique<serve::JournalReader>(cluster.nodes[0], cluster.journal_port,
                                                         120000, rank);
        std::string line;
        if (!reader_->read_line([] { return false; }, &line))
          throw std::runtime_error("research head closed before settings agreement");
        const auto parsed = minijson::parse_document(line);
        const auto& r = parsed.root;
        if (r.at("research_settings").as_int() != 1 || !r.at("heretic_research").as_bool() ||
            !r.at("digest").is_string())
          throw std::runtime_error("research settings agreement mismatch");
        reader_->send_ready(identity);
        if (!reader_->read_line([] { return false; }, &line) ||
            minijson::parse_document(line).root.at("research_ready").as_int() != 1)
          throw std::runtime_error("research settings agreement refused");
      }
    }
    watch_ = std::thread([this] {
      while (!stop_.load()) {
        if (active_.load()) {
          bool dead = reader_ && reader_->rank0_closed();
          if (writer_) dead = dead || writer_->dead_peer() != 0;
          if (rank_ == 0) {
            char byte;
            const auto n = recv(owner_fd_, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
            dead = dead || n == 0 ||
                   (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
          }
          if (dead) {
            std::fprintf(
                stderr, "heretic rank %d: world/owner died during operation; terminating\n", rank_);
            std::fflush(stderr);
            std::_Exit(1);  // Do not block in CUDA teardown after a failed collective.
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }
  ~ResearchWorld() {
    stop_.store(true);
    if (watch_.joinable()) watch_.join();
    if (writer_) writer_->close_peers();
  }
  void active(bool value) { active_.store(value); }
  bool head() const { return rank_ == 0; }
  // Called once by every rank for the same external request. Rank zero has
  // received the owner frame; peers wait for its escaped journal record.
  bool command(std::string* header) {
    if (reader_) {
      std::string line;
      if (!reader_->read_line([] { return false; }, &line)) return false;
      if (line.size() > 512 * 1024)
        throw std::runtime_error("research journal header exceeds budget");
      const auto parsed = minijson::parse_document(line);
      if (!parsed.root.at("request").is_string())
        throw std::runtime_error("invalid research request record");
      *header = std::string(parsed.root.at("request").as_string());
    } else if (writer_) {
      std::string line = "{\"request\":";
      serve::append_json_string(&line, *header);
      writer_->broadcast(line + "}");
    }
    if (header->size() > 65536) throw std::runtime_error("research command exceeds header budget");
    ++sequence_;
    phase_ = 0;
    command_digest_ = serve::config_digest(*header);
    audit("command " + command_digest_);
    return true;
  }
  // The per-command phase and command digest prevent a delayed rank reply
  // from satisfying the next gather. Every rank joins, including empty barriers.
  std::vector<std::string> gather(const std::string& local) {
    ++phase_;
    if (local.size() > kWorldPayloadLimit)
      throw std::runtime_error("research rank payload exceeds budget");
    const std::string stamp =
        std::to_string(sequence_) + "/" + std::to_string(phase_) + "/" + command_digest_ + "\n";
    audit("gather " + std::to_string(phase_));
    if (reader_) {
      if (!reader_->send_payload(stamp + local, kWorldDeadlineMs))
        throw std::runtime_error("research reply failed");
      return {};
    }
    std::vector<std::string> result{local};
    if (writer_) {
      for (auto& [rank, value] :
           writer_->gather_peer_payloads(kWorldPayloadLimit + 128, kWorldDeadlineMs)) {
        if (rank != static_cast<int>(result.size()) || !value.starts_with(stamp))
          throw std::runtime_error("research rank/sequence/phase mismatch");
        result.push_back(value.substr(stamp.size()));
      }
    }
    return result;
  }
  void token(int64_t id) { audit("token " + std::to_string(id)); }
  void barrier() {
    (void)gather({});
    const auto release =
        "research_barrier " + std::to_string(sequence_) + "/" + std::to_string(phase_);
    if (writer_) writer_->broadcast(release);
    if (reader_) {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(kWorldDeadlineMs);
      std::string line;
      if (!reader_->read_line([&] { return std::chrono::steady_clock::now() >= deadline; },
                              &line) ||
          line != release)
        throw std::runtime_error("research load barrier failed");
    }
    audit("barrier " + std::to_string(phase_));
  }

 private:
  void audit(const std::string& event) {
    if (audit_.is_open()) {
      audit_ << sequence_ << ' ' << event << '\n';
      audit_.flush();
      if (!audit_) throw std::runtime_error("research operation log write failed");
    }
  }
  int rank_, owner_fd_;
  int64_t sequence_ = 0, phase_ = 0;
  std::string command_digest_;
  std::unique_ptr<serve::JournalWriter> writer_;
  std::unique_ptr<serve::JournalReader> reader_;
  std::ofstream audit_;
  std::atomic<bool> active_{false}, stop_{false};
  std::thread watch_;
};

template <class T>
inline void append_values(std::string& out, const std::vector<T>& values) {
  if (!values.empty())
    out.append(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}
class TensorReader {
 public:
  explicit TensorReader(const std::string& bytes) : bytes_(bytes) {}
  template <class T>
  std::vector<T> take(size_t count) {
    if (count > (bytes_.size() - offset_) / sizeof(T))
      throw std::runtime_error("truncated research tensors");
    std::vector<T> result(count);
    std::memcpy(result.data(), bytes_.data() + offset_, count * sizeof(T));
    offset_ += count * sizeof(T);
    return result;
  }
  void finish() const {
    if (offset_ != bytes_.size()) throw std::runtime_error("unexpected research tensor bytes");
  }

 private:
  const std::string& bytes_;
  size_t offset_ = 0;
};
}  // namespace dgpp::heretic
