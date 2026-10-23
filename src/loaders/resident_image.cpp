#include "loaders/resident_image.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include "common/log.hpp"

namespace dgpp {

namespace {

constexpr char kMagic[8] = {'D', 'G', 'P', 'P', 'R', 'I', 'M', 'G'};
constexpr size_t kBlobAlign = 4096;

struct Header {
  char magic[8];
  uint32_t version;
  uint32_t layers;
  uint64_t key;
  uint64_t reserved[5];
};
static_assert(sizeof(Header) == 64, "resident image header is 64 bytes");

[[noreturn]] void fail(const std::string& what, const std::string& path) {
  throw std::runtime_error("resident image " + path + ": " + what + " (" +
                           std::strerror(errno) + ")");
}

void pwrite_all(int fd, const void* src, size_t bytes, uint64_t offset,
                const std::string& path) {
  const uint8_t* p = static_cast<const uint8_t*>(src);
  while (bytes > 0) {
    // Linux caps a single pwrite at ~2 GiB; loop in 1 GiB pieces.
    const size_t chunk = bytes < (1u << 30) ? bytes : (1u << 30);
    const ssize_t n = ::pwrite(fd, p, chunk, static_cast<off_t>(offset));
    if (n <= 0) fail("write failed", path);
    p += n;
    bytes -= static_cast<size_t>(n);
    offset += static_cast<uint64_t>(n);
  }
}

void pread_all(int fd, void* dst, size_t bytes, uint64_t offset,
               const std::string& path) {
  uint8_t* p = static_cast<uint8_t*>(dst);
  while (bytes > 0) {
    const size_t chunk = bytes < (1u << 30) ? bytes : (1u << 30);
    const ssize_t n = ::pread(fd, p, chunk, static_cast<off_t>(offset));
    if (n <= 0) fail("read failed or short", path);
    p += n;
    bytes -= static_cast<size_t>(n);
    offset += static_cast<uint64_t>(n);
  }
}

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
uint64_t align_down(uint64_t v, uint64_t a) { return v / a * a; }

bool page_aligned(const void* p) {
  return reinterpret_cast<uintptr_t>(p) % kBlobAlign == 0;
}

}  // namespace

uint64_t ResidentImage::fold(const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t h = 0x9E3779B97F4A7C15ULL ^ bytes;
  size_t i = 0;
  for (; i + 8 <= bytes; i += 8) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    h = (h ^ w) * 0x100000001B3ULL;
  }
  if (i < bytes) {
    uint64_t w = 0;
    std::memcpy(&w, p + i, bytes - i);
    h = (h ^ w) * 0x100000001B3ULL;
  }
  return h;
}

ResidentImage::ResidentImage(const std::string& dir, uint64_t key,
                                   int layers) {
  if (layers <= 0) throw std::invalid_argument("resident image: no layers");
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  char name[32];
  std::snprintf(name, sizeof(name), "%016llx.img",
                static_cast<unsigned long long>(key));
  path_ = (std::filesystem::path(dir) / name).string();
  stem_ = path_.substr(0, path_.size() - 4);
  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd_ < 0) fail("open failed", path_);
  // Best effort: a filesystem without O_DIRECT (tmpfs, some overlays)
  // returns EINVAL and every blob simply goes through fd_.
  direct_fd_ = ::open(path_.c_str(), O_RDWR | O_DIRECT | O_CLOEXEC);
  entries_.assign(static_cast<size_t>(layers), Entry{});

  struct stat st{};
  if (fstat(fd_, &st) != 0) fail("fstat failed", path_);
  const uint64_t table_bytes = sizeof(Header) + sizeof(Entry) * entries_.size();
  bool fresh = static_cast<uint64_t>(st.st_size) < table_bytes;
  if (!fresh) {
    Header h{};
    pread_all(fd_, &h, sizeof(h), 0, path_);
    fresh = std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0 ||
            h.version != kFormatVersion || h.key != key ||
            h.layers != static_cast<uint32_t>(layers);
    if (fresh)
      DGPP_LOG_WARN("resident image {}: header mismatch (version {} key {:#x} "
                    "layers {}) — replacing",
                    path_, h.version, h.key, h.layers);
  }
  if (fresh) {
    write_fresh(key);
  } else {
    pread_all(fd_, entries_.data(), sizeof(Entry) * entries_.size(), sizeof(Header), path_);
    // A published blob must be aligned, outside the table and wholly in
    // the file. Subtract only after checking the offset to avoid wrapping
    // corrupted offset/length pairs. Invalid entries are rebuilt.
    const uint64_t file_bytes = static_cast<uint64_t>(st.st_size);
    for (Entry& e : entries_)
      if (e.bytes != 0 &&
          (e.offset < table_bytes || e.offset % kBlobAlign != 0 || e.offset > file_bytes ||
           e.bytes > file_bytes - e.offset || e.bytes > std::numeric_limits<size_t>::max()))
        e = Entry{};
  }
}

ResidentImage::~ResidentImage() {
  if (direct_fd_ >= 0) ::close(direct_fd_);
  if (fd_ >= 0) ::close(fd_);
}

void ResidentImage::write_fresh(uint64_t key) {
  if (ftruncate(fd_, 0) != 0) fail("truncate failed", path_);
  Header h{};
  std::memcpy(h.magic, kMagic, sizeof(kMagic));
  h.version = kFormatVersion;
  h.layers = static_cast<uint32_t>(entries_.size());
  h.key = key;
  pwrite_all(fd_, &h, sizeof(h), 0, path_);
  for (Entry& e : entries_) e = Entry{};
  pwrite_all(fd_, entries_.data(), sizeof(Entry) * entries_.size(),
             sizeof(Header), path_);
  if (fdatasync(fd_) != 0) fail("fdatasync failed", path_);
}

int ResidentImage::present() const {
  std::lock_guard<std::mutex> lk(mutex_);
  int n = 0;
  for (const Entry& e : entries_) n += e.bytes != 0;
  return n;
}

bool ResidentImage::has_layer(int layer) const {
  std::lock_guard<std::mutex> lk(mutex_);
  return layer >= 0 && layer < layers() &&
         entries_[static_cast<size_t>(layer)].bytes != 0;
}

size_t ResidentImage::layer_bytes(int layer) const {
  std::lock_guard<std::mutex> lk(mutex_);
  if (layer < 0 || layer >= layers() ||
      entries_[static_cast<size_t>(layer)].bytes == 0)
    return 0;
  return entries_[static_cast<size_t>(layer)].bytes;
}

uint64_t ResidentImage::table_offset(int layer) const {
  return sizeof(Header) + sizeof(Entry) * static_cast<uint64_t>(layer);
}

void ResidentImage::write_entry(int layer) const {
  pwrite_all(fd_, &entries_[static_cast<size_t>(layer)], sizeof(Entry),
             table_offset(layer), path_);
}

// Blob bodies: the page-aligned prefix rides O_DIRECT when the buffer and
// the descriptor allow it; the sub-page tail (and everything, when they
// don't) goes buffered. Blob offsets are 4 KiB-aligned by construction, so
// only the buffer's alignment and the byte count decide the split.
void ResidentImage::read_blob(void* dst, size_t bytes,
                                 uint64_t offset) const {
  size_t direct = 0;
  if (direct_fd_ >= 0 && page_aligned(dst))
    direct = align_down(bytes, kBlobAlign);
  if (direct > 0) pread_all(direct_fd_, dst, direct, offset, path_);
  if (direct < bytes)
    pread_all(fd_, static_cast<uint8_t*>(dst) + direct, bytes - direct,
              offset + direct, path_);
}

void ResidentImage::write_blob(const void* src, size_t bytes,
                                  uint64_t offset) const {
  size_t direct = 0;
  if (direct_fd_ >= 0 && page_aligned(src))
    direct = align_down(bytes, kBlobAlign);
  if (direct > 0) pwrite_all(direct_fd_, src, direct, offset, path_);
  if (direct < bytes)
    pwrite_all(fd_, static_cast<const uint8_t*>(src) + direct,
               bytes - direct, offset + direct, path_);
}

bool ResidentImage::verification_enabled() {
  const char* value = std::getenv("DGPP_RESIDENT_CACHE_VERIFY");
  return !value || std::strcmp(value, "0") != 0;
}

void ResidentImage::read_layer(int layer, void* dst, size_t bytes,
                                  bool verify) const {
  // Snapshot the entry under the mutex; the blob itself moves outside it
  // so concurrent restores of different layers stay parallel.
  uint64_t offset = 0;
  uint64_t want = 0;
  uint64_t expect_fold = 0;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (layer < 0 || layer >= layers() ||
        entries_[static_cast<size_t>(layer)].bytes == 0)
      throw std::runtime_error("resident image " + path_ + ": layer " +
                               std::to_string(layer) + " is absent");
    const Entry& e = entries_[static_cast<size_t>(layer)];
    if (e.bytes != bytes)
      throw std::runtime_error(
          "resident image " + path_ + ": layer " + std::to_string(layer) +
          " holds " + std::to_string(e.bytes) + " bytes, the build formula says " +
          std::to_string(bytes) + " (stale image for this loader — delete it)");
    offset = e.offset;
    want = e.bytes;
    expect_fold = e.fold;
  }
  read_blob(dst, want, offset);
  if (verify && fold(dst, want) != expect_fold)
    throw std::runtime_error("resident image " + path_ + ": layer " +
                             std::to_string(layer) +
                             " failed verification (corrupt blob)");
}

bool ResidentImage::try_read_layer(int layer, void* dst, size_t bytes) {
  if (!has_layer(layer)) return false;
  try {
    read_layer(layer, dst, bytes);
    return true;
  } catch (const std::runtime_error& e) {
    DGPP_LOG_WARN("resident image {}: layer {} rejected ({}) — rebuilding from the checkpoint",
                  path_, layer, e.what());
    std::lock_guard<std::mutex> lk(mutex_);
    entries_[static_cast<size_t>(layer)] = Entry{};
    return false;
  }
}

void ResidentImage::write_layer(int layer, const void* src, size_t bytes) {
  if (layer < 0 || layer >= layers() || bytes == 0)
    throw std::invalid_argument("resident image: bad layer/bytes");
  // Reserve the range under the mutex so concurrent captures land on
  // disjoint offsets; the bytes and the entry publish outside/after it.
  // A crash between reserve and publish leaves allocated-but-absent tail
  // bytes (a cache: wasted, never trusted — absent entries rebuild).
  uint64_t offset = 0;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    struct stat st{};
    if (fstat(fd_, &st) != 0) fail("fstat failed", path_);
    offset = align_up(static_cast<uint64_t>(st.st_size), kBlobAlign);
    if (ftruncate(fd_, static_cast<off_t>(offset + bytes)) != 0)
      fail("reserve failed", path_);
  }
  const uint64_t blob_fold = fold(src, bytes);
  write_blob(src, bytes, offset);
  // The blob must be durable before its entry says it exists. (The direct
  // part already is; this covers the buffered tail and the file size.)
  if (fdatasync(fd_) != 0) fail("fdatasync failed", path_);
  {
    std::lock_guard<std::mutex> lk(mutex_);
    Entry& e = entries_[static_cast<size_t>(layer)];
    e.offset = offset;
    e.bytes = bytes;
    e.fold = blob_fold;
    write_entry(layer);
  }
}

std::unique_ptr<ResidentImage> ResidentImage::open_for(const std::string& dir, uint64_t key, int layers,
                                                       uint64_t payload, bool* replaced) {
  if (replaced) *replaced = false;
  auto image = std::make_unique<ResidentImage>(dir, key, layers);
  uint64_t noted = 0;
  if (image->read_note(kPayloadNoteName, &noted, sizeof noted)) {
    if (noted == payload) return image;
    // Another checkpoint's weights under this key: the image, its digest
    // (over the SOURCE) and this note all describe the old bytes.
    const std::filesystem::path stem(image->stem_);
    image.reset();
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
      const std::string name = entry.path().filename().string();
      if (name.rfind(stem.filename().string() + ".", 0) == 0) std::filesystem::remove(entry.path(), ec);
    }
    image = std::make_unique<ResidentImage>(dir, key, layers);
    if (replaced) *replaced = true;
  }
  image->write_note(kPayloadNoteName, &payload, sizeof payload);
  return image;
}

std::string ResidentImage::note_path(const std::string& name) const {
  return stem_ + "." + name;
}

bool ResidentImage::read_note(const std::string& name, void* dst,
                                 size_t bytes) const {
  const std::string path = note_path(name);
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st{};
  bool ok = fstat(fd, &st) == 0 && static_cast<uint64_t>(st.st_size) == bytes;
  if (ok) {
    try {
      pread_all(fd, dst, bytes, 0, path);
    } catch (const std::exception&) {
      ok = false;
    }
  }
  ::close(fd);
  return ok;
}

void ResidentImage::write_note(const std::string& name, const void* src,
                                  size_t bytes) const {
  const std::string path = note_path(name);
  const std::string tmp = path + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                        0644);
  if (fd < 0) fail("note open failed", tmp);
  try {
    pwrite_all(fd, src, bytes, 0, tmp);
    if (fdatasync(fd) != 0) fail("note fdatasync failed", tmp);
  } catch (...) {
    ::close(fd);
    ::unlink(tmp.c_str());
    throw;
  }
  ::close(fd);
  if (::rename(tmp.c_str(), path.c_str()) != 0) fail("note rename failed", path);
}

}  // namespace dgpp
