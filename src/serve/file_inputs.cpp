#include "serve/file_inputs.hpp"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

#include "text/json_grammar.hpp"
#include "text/chat_template.hpp"
#include "text/string_constraint.hpp"

extern char** environ;

namespace dgpp::serve {
namespace {
using V = minijson::Value;
using M = minijson::Member;
namespace fs = std::filesystem;

std::string temporary_directory(const fs::path& parent) {
  std::string path = (parent / "dgpp-files-XXXXXX").string();
  if (!::mkdtemp(path.data())) throw FileInputError("cannot create file-processing directory", "file", 500);
  return path;
}
struct TempDirectory {
  std::string path;
  ~TempDirectory() { std::error_code e; fs::remove_all(path, e); }
};
std::string read_file(const fs::path& path, uint64_t limit) {
  std::error_code ec;
  const auto size = fs::file_size(path, ec);
  if (ec) throw FileInputError("file data is unavailable", "file_id", 404, "file_not_found");
  if (size > limit) throw FileInputError("file exceeds the configured byte limit", "file", 413, "file_too_large");
  std::ifstream in(path, std::ios::binary);
  std::string data(static_cast<size_t>(size), '\0');
  if (!in.read(data.data(), static_cast<std::streamsize>(size))) throw FileInputError("cannot read file", "file", 500);
  return data;
}
void write_file(const fs::path& path, std::string_view data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  out.close();
  if (!out) throw FileInputError("cannot store file", "file", 500);
}
bool valid_id(std::string_view id) {
  return id.starts_with("file-") && id.size() == 37 &&
         id.substr(5).find_first_not_of("0123456789abcdef") == std::string_view::npos;
}
std::string new_id() {
  std::random_device random;
  std::string id = "file-";
  for (int i = 0; i < 4; ++i) {
    uint32_t x = random();
    for (int shift = 28; shift >= 0; shift -= 4) id += "0123456789abcdef"[(x >> shift) & 15];
  }
  return id;
}
V own_value(const V& value) {
  if (value.is_string()) return V::make_owned_string(std::string(value.as_string()));
  if (value.is_array()) {
    std::vector<V> items;
    for (const auto& item : value.items()) items.push_back(own_value(item));
    return V::make_array(std::move(items));
  }
  if (value.is_object()) {
    std::vector<M> members;
    for (const auto& m : value.members()) members.push_back({m.key, own_value(m.value)});
    return V::make_object(std::move(members));
  }
  return value;
}
std::string base64(std::string_view value, uint64_t limit) {
  if (value.starts_with("data:")) {
    size_t comma = value.find(',');
    if (comma == std::string_view::npos || !value.substr(0, comma).ends_with(";base64"))
      throw FileInputError("file_data must be base64 or a base64 data URI", "file_data");
    value.remove_prefix(comma + 1);
  }
  if (value.size() / 4 > limit / 3 + 1) throw FileInputError("file exceeds the configured byte limit", "file_data", 413, "file_too_large");
  if (value.empty() || value.size() % 4) throw FileInputError("file_data is not valid padded base64", "file_data");
  std::array<int, 256> decode;
  decode.fill(-1);
  const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (size_t i = 0; i < alphabet.size(); ++i) decode[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
  std::string out;
  out.reserve(value.size()/4*3);
  for (size_t i = 0; i < value.size(); i += 4) {
    int pad = (value[i+3] == '=') + (value[i+2] == '=');
    if (value[i] == '=' || value[i+1] == '=' || (pad && i+4 != value.size()) ||
        (value[i+2] == '=' && value[i+3] != '=')) throw FileInputError("invalid base64 padding", "file_data");
    uint32_t bits = 0;
    for (int j = 0; j < 4; ++j) {
      int v = value[i+j] == '=' ? 0 : decode[static_cast<unsigned char>(value[i+j])];
      if (v < 0) throw FileInputError("file_data contains invalid base64", "file_data");
      bits = (bits << 6) | static_cast<unsigned>(v);
    }
    if ((pad == 2 && (bits & 0xffff)) || (pad == 1 && (bits & 0xff)))
      throw FileInputError("noncanonical base64 padding", "file_data");
    out += static_cast<char>(bits >> 16);
    if (pad < 2) out += static_cast<char>(bits >> 8);
    if (!pad) out += static_cast<char>(bits);
  }
  if (out.size() > limit) throw FileInputError("file exceeds the configured byte limit", "file_data", 413, "file_too_large");
  return out;
}

std::string parameter(std::string_view header, std::string_view name) {
  std::string result;
  bool found = false;
  size_t pos = header.find(';');
  while (pos != std::string_view::npos && pos < header.size()) {
    ++pos;
    while (pos < header.size() && (header[pos] == ' ' || header[pos] == '\t')) ++pos;
    const auto start = pos;
    while (pos < header.size() && header[pos] != '=' && header[pos] != ';' &&
           header[pos] != '\r' && header[pos] != '\n') ++pos;
    if (pos == header.size() || header[pos] != '=') throw FileInputError("malformed multipart parameter");
    auto key = header.substr(start, pos-start);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.remove_suffix(1);
    ++pos;
    while (pos < header.size() && (header[pos] == ' ' || header[pos] == '\t')) ++pos;
    std::string value;
    if (pos < header.size() && header[pos] == '"') {
      ++pos;
      bool closed = false;
      while (pos < header.size()) {
        char c = header[pos++];
        if (c == '"') { closed = true; break; }
        if (c == '\\' && pos < header.size()) c = header[pos++];
        if (c == '\r' || c == '\n') break;
        value += c;
      }
      if (!closed) throw FileInputError("unterminated multipart parameter");
    } else {
      const auto begin = pos;
      while (pos < header.size() && header[pos] != ';' && header[pos] != '\r' &&
             header[pos] != '\n' && header[pos] != ' ' && header[pos] != '\t') ++pos;
      value = header.substr(begin, pos-begin);
    }
    // MIME parameter names are case-insensitive; their values (including
    // the boundary and multipart field names) retain their original case.
    std::string lower_key(key);
    std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lower_key == name) {
      if (found) throw FileInputError("duplicate multipart parameter '" + std::string(name) + "'");
      found = true; result = std::move(value);
    }
    while (pos < header.size() && (header[pos] == ' ' || header[pos] == '\t')) ++pos;
    if (pos < header.size() && header[pos] != ';') throw FileInputError("malformed multipart parameter suffix");
    if (pos == header.size()) break;
  }
  return result;
}
struct Upload { std::string filename, data, purpose; };
Upload multipart(const HttpRequest& request) {
  const auto* content_type = request.header("content-type");
  const auto media_type = [](std::string_view header) {
    auto value = header.substr(0, header.find(';'));
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.remove_prefix(1);
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.remove_suffix(1);
    std::string lower(value);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    return lower;
  };
  if (!content_type || media_type(*content_type) != "multipart/form-data")
    throw FileInputError("file uploads require multipart/form-data");
  std::string boundary = parameter(*content_type, "boundary");
  if (boundary.empty() || boundary.size() > 70 || boundary.back() == ' ' ||
      std::any_of(boundary.begin(), boundary.end(), [](unsigned char c) { return c < 0x20 || c == 0x7f; }))
    throw FileInputError("invalid multipart boundary");
  boundary = "--" + boundary;
  std::string_view body = request.body;
  size_t pos = 0;
  Upload upload;
  bool have_file = false, have_purpose = false;
  while (true) {
    if (body.substr(pos, boundary.size()) != boundary) throw FileInputError("malformed multipart boundary");
    pos += boundary.size();
    if (body.substr(pos, 2) == "--") {
      if (pos+2 != body.size() && body.substr(pos+2, 2) != "\r\n") throw FileInputError("malformed closing multipart boundary");
      break;
    }
    if (body.substr(pos, 2) != "\r\n") throw FileInputError("malformed multipart part");
    pos += 2;
    size_t head_end = body.find("\r\n\r\n", pos);
    if (head_end == std::string_view::npos || head_end-pos > 16384) throw FileInputError("invalid multipart headers");
    auto headers = body.substr(pos, head_end-pos);
    pos = head_end + 4;
    size_t end = body.find("\r\n" + boundary, pos);
    while (end != std::string_view::npos) {
      const auto after = end + 2 + boundary.size();
      auto suffix = body.substr(after, 2);
      if (suffix == "\r\n" ||
          (suffix == "--" && (after + 2 == body.size() || body.substr(after + 2, 2) == "\r\n"))) break;
      end = body.find("\r\n" + boundary, end + 2);
    }
    if (end == std::string_view::npos) throw FileInputError("unterminated multipart part");
    std::string_view disposition;
    for (size_t at = 0; at < headers.size();) {
      size_t next = headers.find("\r\n", at);
      if (next == std::string_view::npos) next = headers.size();
      auto line = headers.substr(at, next-at);
      const auto colon = line.find(':');
      std::string key(line.substr(0, colon));
      std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
      if (key == "content-disposition" && colon != std::string_view::npos) {
        if (!disposition.empty()) throw FileInputError("duplicate Content-Disposition header");
        disposition = line.substr(colon+1);
      }
      at = next + 2;
    }
    if (media_type(disposition) != "form-data") throw FileInputError("multipart parts require Content-Disposition: form-data");
    const auto name = parameter(disposition, "name");
    if (name == "file") {
      if (have_file) throw FileInputError("duplicate file part");
      have_file = true;
      upload.filename = parameter(disposition, "filename");
      upload.data = body.substr(pos, end-pos);
    } else if (name == "purpose") {
      if (have_purpose) throw FileInputError("duplicate purpose part", "purpose");
      have_purpose = true;
      upload.purpose = body.substr(pos, end-pos);
    } else throw FileInputError("unsupported upload field '" + name + "'", name);
    pos = end + 2;
  }
  if (!have_file || upload.filename.empty()) throw FileInputError("a file with a filename is required");
  if (upload.purpose != "user_data" && upload.purpose != "assistants")
    throw FileInputError("input files require purpose user_data or assistants", "purpose");
  return upload;
}
}  // namespace

FileInputs::FileInputs(FileInputConfig config) : config_(std::move(config)) {
  if (!config_.max_file_bytes || !config_.max_request_bytes || !config_.max_text_bytes ||
      !config_.max_storage_bytes || config_.pdf_timeout_ms <= 0 || config_.workers <= 0)
    throw std::invalid_argument("file input limits and workers must be positive");
  ephemeral_ = config_.directory.empty();
  if (ephemeral_) config_.directory = temporary_directory(fs::temp_directory_path());
  else fs::create_directories(config_.directory);
  for (const auto& entry : fs::directory_iterator(config_.directory)) {
    auto id = entry.path().stem().string();
    if (entry.path().extension() != ".json" || !valid_id(id)) continue;
    const auto json = read_file(entry.path(), 65536);
    auto metadata = own_value(minijson::parse(json).root);
    const auto* bytes = metadata.find("bytes");
    const auto* stored_id = metadata.find("id");
    const auto* filename = metadata.find("filename");
    const auto* purpose = metadata.find("purpose");
    std::error_code ec;
    const auto actual_bytes = fs::file_size(fs::path(config_.directory) / id, ec);
    if (!bytes || bytes->kind() != V::Kind::Int || bytes->as_int() < 0 || ec ||
        static_cast<uint64_t>(bytes->as_int()) != actual_bytes ||
        !stored_id || !stored_id->is_string() || stored_id->as_string() != id ||
        !filename || !filename->is_string() || filename->as_string().empty() ||
        !purpose || !purpose->is_string() || (purpose->as_string() != "user_data" && purpose->as_string() != "assistants") ||
        actual_bytes > UINT64_MAX - stored_bytes_)
      throw std::runtime_error("invalid stored-file metadata: " + id);
    stored_bytes_ += static_cast<uint64_t>(bytes->as_int());
    files_.emplace(id, std::move(metadata));
  }
}
FileInputs::~FileInputs() {
  if (ephemeral_) { std::error_code e; fs::remove_all(config_.directory, e); }
}
bool FileInputs::needed(const HttpRequest& request) {
  if (request.body.find("file") == std::string::npos && request.body.find('\\') == std::string::npos) return false;
  try {
    auto parsed = minijson::parse_document(request.body);
    const auto* messages = parsed.root.find("messages");
    if (!messages || !messages->is_array()) return false;
    for (const auto& msg : messages->items()) {
      const auto* content = msg.find("content");
      if (!content || !content->is_array()) continue;
      for (const auto& part : content->items()) {
        const auto* type = part.find("type");
        if (type && type->is_string() && type->as_string() == "file") return true;
      }
    }
  } catch (const std::exception&) { /* existing API parser reports malformed JSON */ }
  return false;
}

std::string FileInputs::resolve(const V& file, std::string* filename) {
  const auto* id = file.find("file_id");
  const auto* data = file.find("file_data");
  if (!file.is_object() || bool(id) == bool(data)) throw FileInputError("file requires exactly one of file_id or file_data");
  if (const auto* name = file.find("filename")) {
    if (!name->is_string()) throw FileInputError("filename must be a string", "filename");
    *filename = name->as_string();
  }
  if (id) {
    if (!id->is_string() || !valid_id(id->as_string())) throw FileInputError("unknown file_id", "file_id", 404, "file_not_found");
    std::lock_guard lock(mutex_);
    auto it = files_.find(std::string(id->as_string()));
    if (it == files_.end()) throw FileInputError("unknown file_id", "file_id", 404, "file_not_found");
    *filename = it->second.find("filename")->as_string();
    return read_file(fs::path(config_.directory) / it->first, config_.max_file_bytes);
  }
  if (!data->is_string()) throw FileInputError("file_data must be a base64 string", "file_data");
  return base64(data->as_string(), config_.max_file_bytes);
}

std::string FileInputs::extract(const std::string& data, const std::string& filename) {
  if (stopping_.load(std::memory_order_relaxed))
    throw FileInputError("server is shutting down", "file", 503, "server_shutdown");
  if (!data.starts_with("%PDF-")) {
    // Text/code files are a DGPP extension; binary office/image files are
    // never fed to the tokenizer as though their bytes were document text.
    const std::string extension = fs::path(filename).extension().string();
    static const std::string extensions = " .txt .md .markdown .json .jsonl .csv .tsv .html .xml .yaml .yml .log .py .js .ts .tsx .jsx .c .cpp .h .hpp .rs .go .sh .sql .toml .ini ";
    if (extension.empty() || extensions.find(" " + extension + " ") == std::string::npos)
      throw FileInputError("supported file inputs are PDF and UTF-8 text/code files", "file");
    if (data.size() > config_.max_text_bytes) throw FileInputError("extracted text exceeds max_text_bytes", "file", 413);
    if (data.find('\0') != std::string::npos) throw FileInputError("text file contains binary data", "file");
    std::string decoded; bool complete = false;
    if (!text::decode_json_string_prefix(text::Value::string_value(data).to_json(false), &decoded, &complete) || !complete)
      throw FileInputError("text file must contain valid UTF-8", "file");
    return data;
  }
  TempDirectory temp{temporary_directory(config_.directory)};
  const auto input = fs::path(temp.path) / "input.pdf";
  const auto output = fs::path(temp.path) / "output.txt";
  write_file(input, data);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
  std::vector<std::string> args{config_.pdf_command, "-enc", "UTF-8", "-layout", input.string(), output.string()};
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  argv.push_back(nullptr);
  pid_t child = -1;
  const int spawned = posix_spawnp(&child, argv[0], &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (spawned) throw FileInputError("PDF extraction requires pdftotext (poppler-utils) or a configured pdf_command", "file", 503, "pdf_extractor_unavailable");
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.pdf_timeout_ms);
  int status = 0;
  while (true) {
    const auto waited = waitpid(child, &status, WNOHANG);
    if (waited == child) break;
    if (waited < 0) {
      if (errno == EINTR) continue;
      throw FileInputError("cannot wait for PDF extractor", "file", 500);
    }
    std::error_code ec;
    auto size = fs::file_size(output, ec);
    const bool large = !ec && size > config_.max_text_bytes;
    const bool stopping = stopping_.load(std::memory_order_relaxed);
    if (stopping || large || std::chrono::steady_clock::now() >= deadline) {
      kill(child, SIGKILL);
      while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
      if (stopping) throw FileInputError("server is shutting down", "file", 503, "server_shutdown");
      throw FileInputError(large ? "PDF text exceeds max_text_bytes" : "PDF extraction timed out", "file", large ? 413 : 408);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status)) throw FileInputError("PDF extraction failed; the PDF may be malformed or encrypted", "file");
  std::string result = read_file(output, config_.max_text_bytes);
  if (result.find_first_not_of(" \t\r\n\f") == std::string::npos)
    throw FileInputError("PDF has no extractable text; scanned pages require OCR or a vision model", "file", 400, "pdf_text_unavailable");
  return result;
}

HttpRequest FileInputs::prepare(HttpRequest request) {
  minijson::ParseResult parsed;
  try {
    parsed = minijson::parse_document(request.body);
  } catch (const std::exception& e) {
    throw FileInputError(std::string("invalid JSON body: ") + e.what(), "body");
  }
  const auto* messages = parsed.root.find("messages");
  if (!messages || !messages->is_array()) throw FileInputError("messages must be an array", "messages");
  std::vector<V> normalized;
  uint64_t bytes = 0, text_bytes = 0;
  for (size_t i = 0; i < messages->items().size(); ++i) {
    const auto& msg = messages->items()[i];
    const auto* content = msg.find("content");
    if (!content || !content->is_array()) { normalized.push_back(msg); continue; }
    std::vector<V> parts;
    for (size_t j = 0; j < content->items().size(); ++j) {
      const auto& part = content->items()[j];
      const auto* type = part.find("type");
      if (!type || !type->is_string() || type->as_string() != "file") { parts.push_back(part); continue; }
      std::string path = "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "]";
      try {
        const auto* role = msg.find("role");
        if (!role || role->as_string() != "user") throw FileInputError("file parts are only supported on user messages");
        if (part.find("prompt_cache_breakpoint")) throw FileInputError("explicit cache breakpoints are not supported", "prompt_cache_breakpoint");
        const auto* file = part.find("file");
        if (!file) throw FileInputError("file object is required");
        std::string filename;
        auto data = resolve(*file, &filename);
        bytes += data.size();
        if (bytes > config_.max_request_bytes) throw FileInputError("combined files exceed max_request_bytes", "file", 413);
        auto text = extract(data, filename);
        text_bytes += text.size();
        if (text_bytes > config_.max_text_bytes) throw FileInputError("combined extracted text exceeds max_text_bytes", "file", 413);
        parts.push_back(V::make_object({{"type", V::make_string("text")}, {"text", V::make_owned_string(
            "\n[Document: " + filename + "]\n" + text + "\n[End document]\n")}}));
      } catch (const FileInputError& e) { throw FileInputError(e.what(), path + ".file" + (e.param == "file" ? "" : "." + e.param), e.status, e.code); }
    }
    std::vector<M> members;
    for (const auto& m : msg.members()) members.push_back(m.key == "content" ? M{m.key, V::make_array(parts)} : m);
    normalized.push_back(V::make_object(std::move(members)));
  }
  std::vector<M> body;
  for (const auto& m : parsed.root.members()) body.push_back(m.key == "messages" ? M{m.key, V::make_array(normalized)} : m);
  request.body = text::json_text_of(V::make_object(std::move(body)));
  return request;
}

FileResponse FileInputs::route(const HttpRequest& request) {
  std::lock_guard lock(mutex_);
  if (stopping_.load(std::memory_order_relaxed))
    throw FileInputError("server is shutting down", "file", 503, "server_shutdown");
  if (request.path == "/v1/files" && request.method == "POST") {
    auto upload = multipart(request);
    if (upload.data.size() > config_.max_file_bytes) throw FileInputError("file exceeds max_file_bytes", "file", 413);
    if (upload.filename.size() > 1024) throw FileInputError("filename is too long", "filename");
    if (upload.data.size() > config_.max_storage_bytes - std::min(stored_bytes_, config_.max_storage_bytes))
      throw FileInputError("file storage capacity exhausted", "file", 413, "file_storage_full");
    const auto id = new_id();
    auto metadata = V::make_object({{"id", V::make_owned_string(id)}, {"object", V::make_string("file")},
      {"bytes", V::make_int(static_cast<int64_t>(upload.data.size()))}, {"created_at", V::make_int(std::time(nullptr))},
      {"filename", V::make_owned_string(upload.filename)}, {"purpose", V::make_owned_string(upload.purpose)},
      {"status", V::make_string("processed")}, {"status_details", V{}}});
    const auto body = text::json_text_of(metadata);
    write_file(fs::path(config_.directory) / id, upload.data);
    write_file(fs::path(config_.directory) / (id + ".json.tmp"), body);
    fs::rename(fs::path(config_.directory) / (id + ".json.tmp"), fs::path(config_.directory) / (id + ".json"));
    stored_bytes_ += upload.data.size(); files_.emplace(id, metadata);
    return {200, "application/json", body};
  }
  if (request.path == "/v1/files" && request.method == "GET") {
    std::vector<V> files;
    for (const auto& [_, metadata] : files_) files.push_back(metadata);
    return {200, "application/json", text::json_text_of(V::make_object({{"object", V::make_string("list")},
      {"data", V::make_array(std::move(files))}, {"has_more", V::make_bool(false)}}))};
  }
  if (request.path == "/v1/files") throw FileInputError("use GET or POST for files", "file", 405);
  const bool content = request.path.ends_with("/content");
  const std::string id = request.path.substr(10, request.path.size()-10-(content ? 8 : 0));
  auto it = files_.find(id);
  if (!valid_id(id) || it == files_.end()) throw FileInputError("unknown file_id", "file_id", 404, "file_not_found");
  if (request.method == "GET") {
    if (content) return {200, "application/octet-stream", read_file(fs::path(config_.directory)/id, config_.max_file_bytes)};
    return {200, "application/json", text::json_text_of(it->second)};
  }
  if (request.method == "DELETE" && !content) {
    fs::remove(fs::path(config_.directory) / (id + ".json"));
    fs::remove(fs::path(config_.directory) / id);
    stored_bytes_ -= static_cast<uint64_t>(it->second.find("bytes")->as_int());
    files_.erase(it);
    return {200, "application/json", text::json_text_of(V::make_object({{"id", V::make_owned_string(id)},
      {"object", V::make_string("file")}, {"deleted", V::make_bool(true)}}))};
  }
  throw FileInputError("unsupported file endpoint method", "file", 405);
}
}  // namespace dgpp::serve
