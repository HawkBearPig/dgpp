#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"

namespace dgpp {

class SafetensorsFile;

struct TensorInfo {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  uint64_t data_begin = 0;  // byte offset into the FILE (header included)
  uint64_t data_end = 0;
  const void* data = nullptr;  // mapped pointer (file lifetime)
  const SafetensorsFile* owner = nullptr;  // the shard the bytes live in

  size_t numel() const {
    size_t n = 1;
    for (const auto d : shape) {
      if (d < 0) throw std::runtime_error("safetensors: negative tensor shape dimension");
      if (static_cast<uint64_t>(d) > std::numeric_limits<size_t>::max() ||
          (n != 0 && static_cast<uint64_t>(d) > std::numeric_limits<size_t>::max() / n))
        throw std::runtime_error("safetensors: tensor shape element count overflows");
      n *= static_cast<size_t>(d);
    }
    return n;
  }
  size_t nbytes() const {
    if (data_end < data_begin || data_end - data_begin > std::numeric_limits<size_t>::max())
      throw std::runtime_error("safetensors: invalid tensor byte range");
    return static_cast<size_t>(data_end - data_begin);
  }
};

class SafetensorsFile {
 public:
  ~SafetensorsFile() { close_mapping(/*drop_page_cache=*/false); }
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

  // Tears the mapping down and, when asked, evicts the file's pages from
  // the page cache (POSIX_FADV_DONTNEED — only effective AFTER munmap: the
  // kernel will not drop pages a live mapping still references). A
  // resident loader calls this once every byte it needs has been copied
  // out: a 70 GB checkpoint left cached next to an 80 GB resident model
  // pins the box at its memory watermark for the whole run, and the
  // kernel then swaps the process's own cold pages out from under it
  // (2026-09-02: ~10 ms swap-in faults in the decode loop). Every tensor
  // view handed out from this file dangles afterwards — callers drop
  // theirs first.
  void close_mapping(bool drop_page_cache) {
    if (map_ && map_ != MAP_FAILED) {
      munmap(map_, map_len_);
      map_ = nullptr;
    }
    if (fd_ >= 0) {
      if (drop_page_cache) posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
      ::close(fd_);
      fd_ = -1;
    }
    tensors_.clear();
  }
  bool mapped() const { return map_ != nullptr && map_ != MAP_FAILED; }

  // Streaming hints for a one-pass reader (the resident loader): ask the
  // kernel to read the tensor's pages ahead as one sequential burst
  // (MADV_WILLNEED — a page-fault walk otherwise serves 4 KB faults with
  // fault-around, a fraction of the NVMe's rate), and, once copied out,
  // drop them from both the private mapping and the page cache
  // (MADV_DONTNEED + POSIX_FADV_DONTNEED). Without the drop, a 300 GB
  // checkpoint read through the cache beside an 80 GB resident model
  // drives every fault through direct reclaim (2026-09-02: allocstall
  // ~1300/s for the whole ~260 s load). Page-rounded; the neighbours'
  // boundary pages are refaulted if still needed. Both are advisory.
  void prefetch(const TensorInfo& t) const {
    if (!mapped() || t.data_end <= t.data_begin) return;
    const auto [begin, len] = page_span(t);
    madvise(map_ + begin, len, MADV_WILLNEED);
  }
  void discard(const TensorInfo& t) const {
    if (!mapped() || t.data_end <= t.data_begin) return;
    const auto [begin, len] = page_span(t);
    madvise(map_ + begin, len, MADV_DONTNEED);
    posix_fadvise(fd_, static_cast<off_t>(begin), static_cast<off_t>(len),
                  POSIX_FADV_DONTNEED);
  }

  static std::unique_ptr<SafetensorsFile> open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
      throw std::runtime_error(std::string("cannot open ") + path + ": " +
                               strerror(errno));
    struct stat st{};
    if (fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("fstat failed for " + path);
    }
    size_t len = static_cast<size_t>(st.st_size);
    void* map = mmap(nullptr, len ? len : 1, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("mmap failed for " + path);
    }
    std::unique_ptr<SafetensorsFile> f(new SafetensorsFile());
    f->path_ = path;
    f->fd_ = fd;
    f->map_ = static_cast<uint8_t*>(map);
    f->map_len_ = len;
    f->parse_header();
    return f;
  }

  const TensorInfo* find(std::string_view name) const {
    auto it = tensors_.find(std::string(name));
    return it == tensors_.end() ? nullptr : it->second.get();
  }

  const TensorInfo& at(std::string_view name) const {
    const TensorInfo* t = find(name);
    if (!t)
      throw std::runtime_error("tensor not found: " + std::string(name) +
                               " in " + path_);
    return *t;
  }

  // All tensors whose names match substring (cold-path convenience).
  std::vector<const TensorInfo*> grep(std::string_view substr) const {
    std::vector<const TensorInfo*> out;
    for (const auto& [n, t] : tensors_)
      if (n.find(substr) != std::string::npos) out.push_back(t.get());
    return out;
  }

  size_t tensor_count() const { return tensors_.size(); }
  const uint8_t* map_base() const { return map_; }
  size_t map_size() const { return map_len_; }
  const std::string& path() const { return path_; }
  const minijson::Value& header_meta() const { return meta_; }
  // FNV-1a of the raw header JSON: the shard's identity for cache keys
  // (names, dtypes, shapes, offsets — a swapped checkpoint changes it).
  uint64_t header_fold() const { return header_fold_; }
  // FNV-1a over three 64 KiB windows of the payload (its start, middle and
  // end): the shard's WEIGHT identity, for the resident image's payload
  // note. A refined release that re-ships the same tensors in the same
  // shards with other values keeps header_fold() and changes this
  // (DeepSeek-V4.1-Flash-1017 against DeepSeek-V4.1-Flash, 2026-10-18:
  // byte-identical config and headers, every shard's payload different).
  // Three page-ins per shard; nothing else is read.
  uint64_t payload_sample_fold() const {
    uint64_t hlen = 0;
    std::memcpy(&hlen, map_, 8);
    const size_t begin = 8 + static_cast<size_t>(hlen);
    const size_t len = map_len_ > begin ? map_len_ - begin : 0;
    constexpr size_t kWindow = 64 * 1024;
    const size_t starts[3] = {0, len / 2, len > kWindow ? len - kWindow : 0};
    uint64_t h = 1469598103934665603ull;
    for (const size_t s0 : starts) {
      const size_t n = std::min(kWindow, len - std::min(len, s0));
      const uint8_t* w = map_ + begin + s0;
      for (size_t i = 0; i < n; ++i) h = (h ^ w[i]) * 1099511628211ull;
    }
    return h;
  }

  // Header-truth iteration (bind-check tooling): visits every tensor of this
  // shard. Fn takes const TensorInfo&.
  template <typename Fn>
  void for_each(Fn&& fn) const {
    for (const auto& kv : tensors_) fn(*kv.second);
  }

 private:
  SafetensorsFile() = default;

  void parse_header() {
    if (map_len_ < 8)
      throw std::runtime_error(path_ + ": too small for safetensors");
    uint64_t hlen;
    std::memcpy(&hlen, map_, 8);
    if (hlen > map_len_ - 8) throw std::runtime_error(path_ + ": header length overruns file");
    std::string_view json(reinterpret_cast<const char*>(map_ + 8), hlen);
    header_fold_ = 1469598103934665603ull;
    for (const char c : json)
      header_fold_ = (header_fold_ ^ static_cast<uint8_t>(c)) * 1099511628211ull;
    auto parsed = minijson::parse(json);
    const minijson::Value& root = parsed.root;
    if (!root.is_object())
      throw std::runtime_error(path_ + ": safetensors header not an object");
    uint64_t data_start = 8 + hlen;
    for (const auto& m : root.members()) {
      if (m.key == "__metadata__") {
        meta_ = m.value;  // Value is copyable now
        continue;
      }
      const minijson::Value& desc = m.value;
      auto t = std::make_unique<TensorInfo>();
      t->name = m.key;
      auto ds = desc.at("dtype").as_string();
      auto dt = dtype_from_string(ds);
      if (!dt)
        throw std::runtime_error(path_ + ": unknown dtype " + std::string(ds));
      t->dtype = *dt;
      const auto nonnegative_integer = [&](const minijson::Value& value,
                                           const char* field) -> uint64_t {
        if (value.kind() != minijson::Value::Kind::Int || value.as_int() < 0)
          throw std::runtime_error(path_ + ": tensor " + m.key + " " + field +
                                   " must contain nonnegative integers");
        return static_cast<uint64_t>(value.as_int());
      };
      const minijson::Value& shape = desc.at("shape");
      if (!shape.is_array())
        throw std::runtime_error(path_ + ": tensor " + m.key + " shape is not an array");
      for (const auto& d : shape.items())
        t->shape.push_back(static_cast<int64_t>(nonnegative_integer(d, "shape")));
      const minijson::Value& offs = desc.at("data_offsets");
      if (!offs.is_array() || offs.items().size() != 2)
        throw std::runtime_error(path_ + ": tensor " + m.key +
                                 " data_offsets must contain two integers");
      const uint64_t begin = nonnegative_integer(offs.items()[0], "data_offsets");
      const uint64_t end = nonnegative_integer(offs.items()[1], "data_offsets");
      // Validate payload-relative integers before adding the header offset.
      // Subtraction is safe because the header length was checked above.
      if (begin > end || end > map_len_ - data_start)
        throw std::runtime_error(path_ + ": tensor " + m.key + " has invalid data_offsets");
      const size_t elements = t->numel();
      const size_t width = dtype_size(t->dtype);
      if (elements > std::numeric_limits<size_t>::max() / width || elements * width != end - begin)
        throw std::runtime_error(path_ + ": tensor " + m.key + " shape/dtype byte count mismatch");
      t->data_begin = data_start + begin;
      t->data_end = data_start + end;
      t->data = map_ + t->data_begin;
      t->owner = this;
      tensors_[t->name] = std::move(t);
    }
  }

  std::pair<size_t, size_t> page_span(const TensorInfo& t) const {
    static const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t begin = (t.data_begin / page) * page;
    size_t end = ((t.data_end + page - 1) / page) * page;
    if (end > map_len_) end = map_len_;
    return {begin, end - begin};
  }

  std::string path_;
  int fd_ = -1;
  uint8_t* map_ = nullptr;
  size_t map_len_ = 0;
  uint64_t header_fold_ = 0;
  std::unordered_map<std::string, std::unique_ptr<TensorInfo>> tensors_;
  minijson::Value meta_{};
};

}  // namespace dgpp
