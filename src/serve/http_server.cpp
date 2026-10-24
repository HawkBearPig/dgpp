#include "serve/http_server.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

#include "common/log.hpp"
#include "serve/json_out.hpp"

namespace dgpp::serve {

namespace {

constexpr size_t kMaxHeaderBytes = 16 * 1024;
constexpr size_t kReadChunk = 16 * 1024;

bool set_nonblock(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return false;
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

std::string to_lower(std::string_view s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return out;
}

const char* status_text(int code) {
  switch (code) {
    case 200: return "OK";
    case 201: return "Created";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Status";
  }
}

// One chunked-encoding frame around `data`.
std::string chunk_frame(std::string_view data) {
  char hex[16];
  const int n = std::snprintf(hex, sizeof(hex), "%zx", data.size());
  std::string out;
  out.reserve(data.size() + static_cast<size_t>(n) + 8);
  out.append(hex, static_cast<size_t>(n));
  out.append("\r\n", 2);
  out.append(data);
  out.append("\r\n", 2);
  return out;
}

}  // namespace

struct HttpWakeup::State {
  int fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  State() {
    if (fd < 0) throw std::runtime_error("http: eventfd");
  }
  ~State() { ::close(fd); }
};

void HttpWakeup::operator()() const noexcept {
  if (!state_) return;
  const uint64_t one = 1;
  // EAGAIN means the counter is already full: a wake is pending. The fd
  // cannot close while this handle lives. Never block an engine callback.
  while (::write(state_->fd, &one, sizeof(one)) < 0 && errno == EINTR) {
  }
}

// ---------------------------------------------------------------------------
// HttpRequest
// ---------------------------------------------------------------------------

const std::string* HttpRequest::header(std::string_view name) const {
  const std::string key = to_lower(name);
  for (const auto& [k, v] : headers)
    if (k == key) return &v;
  return nullptr;
}

// ---------------------------------------------------------------------------
// Conn — the one mutable connection record. Members are engine-internal
// (the writer and the server are the only touchers, both on the epoll
// thread), so plain fields with no access ceremony.
// ---------------------------------------------------------------------------

struct Conn {
  std::chrono::steady_clock::time_point last_stream_write;
  int fd = -1;
  bool keep_alive = true;
  bool stream = false;       // chunked SSE mode
  bool closing = false;      // close after the outbuf drains
  bool gone = false;         // peer closed / write failed
  bool busy = false;         // a dispatched request the handler has not
                             // answered yet: the next pipelined request
                             // waits in `in` (one record per connection,
                             // one tag — the fuzzer's third find)
  uint64_t stream_tag = 0;
  std::string in;            // buffered input
  std::string out;           // pending output
  size_t flushed = 0;        // out bytes already sent
  HttpResponseWriter writer; // stable for the conn's lifetime; handlers
                             // may hold it across idle() passes

  ~Conn() {
    if (fd >= 0) ::close(fd);
  }
};

// ---------------------------------------------------------------------------
// HttpResponseWriter
// ---------------------------------------------------------------------------

void HttpResponseWriter::respond(int status, std::string_view content_type,
                                 std::string body) {
  if (conn_ == nullptr || conn_->fd < 0) return;
  Conn& c = *conn_;
  c.busy = false;  // the request is answered: the next one may be read
  char head[256];
  const int n = std::snprintf(
      head, sizeof(head),
      "HTTP/1.1 %d %s\r\nContent-Type: %.*s\r\nContent-Length: %zu\r\n"
      "Connection: %s\r\n\r\n",
      status, status_text(status), static_cast<int>(content_type.size()),
      content_type.data(), body.size(),
      c.keep_alive ? "keep-alive" : "close");
  c.out.append(head, static_cast<size_t>(n));
  c.out.append(body);
}

void HttpResponseWriter::begin_stream(std::string_view content_type) {
  if (conn_ == nullptr || conn_->fd < 0) return;
  Conn& c = *conn_;
  c.stream = true;
  c.last_stream_write = std::chrono::steady_clock::now();
  char head[256];
  const int n = std::snprintf(
      head, sizeof(head),
      "HTTP/1.1 200 OK\r\nContent-Type: %.*s\r\nCache-Control: no-cache\r\n"
      "Transfer-Encoding: chunked\r\nConnection: %s\r\n\r\n",
      static_cast<int>(content_type.size()), content_type.data(),
      c.keep_alive ? "keep-alive" : "close");
  c.out.append(head, static_cast<size_t>(n));
}

bool HttpResponseWriter::write_event(std::string_view data) {
  if (conn_ == nullptr || conn_->fd < 0 || conn_->gone || conn_->closing)
    return false;
  Conn& c = *conn_;
  // SSE framing rides inside the chunk payload: "data: ...\n\n".
  std::string frame;
  frame.reserve(data.size() + 8);
  frame.append("data: ").append(data).append("\n\n", 2);
  c.out.append(chunk_frame(frame));
  c.last_stream_write = std::chrono::steady_clock::now();
  return true;
}

bool HttpResponseWriter::write_comment(std::string_view comment) {
  if (conn_ == nullptr || conn_->fd < 0 || conn_->gone || conn_->closing || !conn_->stream)
    return false;
  if (comment.find_first_of("\r\n") != std::string_view::npos)
    throw std::invalid_argument("http: SSE comment must be a single line");
  Conn& c = *conn_;
  std::string frame;
  frame.reserve(comment.size() + 4);
  frame.append(": ").append(comment).append("\n\n");
  c.out.append(chunk_frame(frame));
  c.last_stream_write = std::chrono::steady_clock::now();
  return true;
}

void HttpResponseWriter::ping_if_idle(int interval_s) {
  if (interval_s <= 0 || conn_ == nullptr || !conn_->stream || conn_->flushed < conn_->out.size())
    return;
  if (std::chrono::steady_clock::now() - conn_->last_stream_write >=
      std::chrono::seconds(interval_s))
    write_comment("keep-alive");
}

void HttpResponseWriter::end_stream() {
  if (conn_ == nullptr || conn_->fd < 0) return;
  conn_->out.append("0\r\n\r\n", 5);
  conn_->stream = false;
  conn_->busy = false;
}

void HttpResponseWriter::set_stream_tag(uint64_t tag) {
  if (conn_ != nullptr) conn_->stream_tag = tag;
}
uint64_t HttpResponseWriter::stream_tag() const {
  return conn_ != nullptr ? conn_->stream_tag : 0;
}
bool HttpResponseWriter::stream_open() const {
  return conn_ != nullptr && conn_->stream;
}
bool HttpResponseWriter::client_gone() const {
  return conn_ == nullptr || conn_->gone || conn_->fd < 0;
}

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------

HttpServer::HttpServer(uint16_t port, HttpHandler* handler, int max_connections,
                       const std::string& bind_host, int64_t max_body_bytes, bool wake_on_output,
                       int poll_timeout_ms)
    : port_(port),
      handler_(handler),
      max_connections_(max_connections),
      max_body_bytes_(static_cast<size_t>(max_body_bytes)),
      poll_timeout_ms_(poll_timeout_ms) {
  if (poll_timeout_ms < 1) throw std::invalid_argument("http: poll timeout must be positive");
  if (max_body_bytes < 1 ||
      static_cast<uint64_t>(max_body_bytes) > std::string{}.max_size() - kMaxHeaderBytes)
    throw std::invalid_argument("http: max_body_bytes must be positive and fit the request buffer");
  if (handler_ == nullptr)
    throw std::invalid_argument("HttpServer: handler must not be null");
  if (wake_on_output || handler_->needs_work_wakeup())
    wakeup_.state_ = std::make_shared<HttpWakeup::State>();
  try {
    in_addr bind_address{};
    if (::inet_pton(AF_INET, bind_host.c_str(), &bind_address) != 1)
      throw std::invalid_argument("http: bind host must be an IPv4 address");
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("http: socket");
    int yes = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr = bind_address;
    addr.sin_port = htons(port_);
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      ::close(listen_fd_);
      listen_fd_ = -1;
      throw std::runtime_error("http: bind port " + std::to_string(port_) + " — " +
                               std::strerror(errno));
    }
    if (::listen(listen_fd_, 64) < 0) {
      ::close(listen_fd_);
      listen_fd_ = -1;
      throw std::runtime_error("http: listen");
    }
    if (port_ == 0) {  // ephemeral: report what the kernel picked
      sockaddr_in bound{};
      socklen_t len = sizeof(bound);
      if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound), &len) == 0)
        port_ = ntohs(bound.sin_port);
    }
    if (!set_nonblock(listen_fd_)) throw std::runtime_error("http: listener nonblock");
    epoll_fd_ = ::epoll_create1(0);
    if (epoll_fd_ < 0) throw std::runtime_error("http: epoll_create1");
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd_;
    if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev) < 0)
      throw std::runtime_error("http: epoll add listener");
    if (wakeup_) {
      ev.data.fd = wakeup_.state_->fd;
      if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, ev.data.fd, &ev) < 0)
        throw std::runtime_error("http: epoll add wakeup");
    }
    handler_->set_wakeup(wake_on_output ? wakeup_ : HttpWakeup{});
    handler_->set_work_wakeup(wakeup_);
  } catch (...) {
    if (listen_fd_ >= 0) ::close(listen_fd_);
    if (epoll_fd_ >= 0) ::close(epoll_fd_);
    throw;
  }
}

HttpServer::~HttpServer() {
  if (listen_fd_ >= 0) ::close(listen_fd_);
  if (epoll_fd_ >= 0) ::close(epoll_fd_);
}

void HttpServer::stop() {
  stopping_.store(true, std::memory_order_release);
  wakeup_();
}

void HttpServer::serve() {
  std::vector<epoll_event> events(64);
  while (!stopping_.load(std::memory_order_acquire)) {
    const int n =
        ::epoll_wait(epoll_fd_, events.data(), static_cast<int>(events.size()), poll_timeout_ms_);
    if (n < 0) {
      if (errno == EINTR) continue;
      DGPP_LOG_ERROR("http: epoll_wait failed — {}", std::strerror(errno));
      break;
    }
    for (int i = 0; i < n; ++i) {
      const int fd = events[i].data.fd;
      const uint32_t evs = events[i].events;
      if (wakeup_ && fd == wakeup_.state_->fd) {
        uint64_t count;
        // One read clears the accumulated counter. A concurrent later write
        // remains readable for the next pass; do not spin draining producers.
        while (::read(fd, &count, sizeof(count)) < 0 && errno == EINTR) {
        }
        continue;
      }
      if (fd == listen_fd_) {
        accept_new();
        continue;
      }
      const auto it = conns_.find(fd);
      if (it == conns_.end()) continue;  // swept in an earlier batch
      Conn& c = *it->second;
      if (evs & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) {
        // The client's goodbye: notify and close. (A half-close could
        // still receive our queued bytes, but treating RDHUP as final
        // is the honest SSE contract — the service cancels the request.)
        close_conn(c, /*notify_disconnect=*/true);
        continue;
      }
      if (evs & EPOLLIN) on_readable(c);
      if (c.fd < 0) continue;  // closed inside on_readable
      if (evs & EPOLLOUT) on_writable(c);
    }
    // The loop pass's tail: the service's SSE drain (idle()) queues
    // events, so flush everything that gained bytes. close_conn only
    // MARKS dead connections (fd = -1); the sweep below is the single
    // place nodes leave the map — no iterator is ever invalidated.
    handler_->idle();
    // A request that was waiting behind one the handler has just answered
    // (pipelined input is parsed one request at a time): read it now — no
    // epoll event will announce bytes that already sit in the buffer.
    for (auto& [fd, c] : conns_) {
      if (c->fd >= 0 && !c->busy && !c->closing && !c->in.empty())
        (void)process_requests(*c);
    }
    for (auto& [fd, c] : conns_) {
      if (c->fd >= 0 && !c->out.empty()) flush_out(*c);
    }
    for (auto it = conns_.begin(); it != conns_.end();) {
      if (it->second->fd < 0)
        it = conns_.erase(it);
      else
        ++it;
    }
  }
  for (auto& [fd, c] : conns_) close_conn(*c, /*notify_disconnect=*/false);
  conns_.clear();
}

void HttpServer::accept_new() {
  while (true) {
    const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_NONBLOCK);
    if (fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return;
      if (errno == EINTR) continue;
      DGPP_LOG_ERROR("http: accept failed — {}", std::strerror(errno));
      return;
    }
    if (static_cast<int>(conns_.size()) >= max_connections_) {
      const std::string body = "{\"error\":{\"message\":\"connection limit reached\","
                               "\"type\":\"server_error\",\"param\":null,\"code\":\"overloaded\"}}";
      const std::string busy = "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                               "Content-Length: " + std::to_string(body.size()) +
                               "\r\nConnection: close\r\n\r\n" + body;
      (void)!::send(fd, busy.data(), busy.size(), MSG_NOSIGNAL);
      ::close(fd);
      continue;
    }
    int yes = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    auto c = std::make_unique<Conn>();
    c->fd = fd;
    c->writer.attach(c.get());  // the writer outlives this request
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLRDHUP;
    ev.data.fd = fd;
    if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
      DGPP_LOG_ERROR("http: epoll add conn — {}", std::strerror(errno));
      ::close(fd);
      continue;
    }
    conns_[fd] = std::move(c);
  }
}

void HttpServer::on_readable(Conn& c) {
  char buf[kReadChunk];
  while (!c.gone) {
    const ssize_t got = ::recv(c.fd, buf, sizeof(buf), 0);
    if (got < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return;
      if (errno == EINTR) continue;
      close_conn(c, /*notify_disconnect=*/true);
      return;
    }
    if (got == 0) {  // orderly goodbye from the peer
      close_conn(c, /*notify_disconnect=*/true);
      return;
    }
    // Once a refusal is queued, discard later input without parsing or
    // buffering it. Leaving it unread can make close() reset the socket
    // and discard response bytes that are still in the kernel send queue.
    if (c.closing) continue;
    c.in.append(buf, static_cast<size_t>(got));
    if (!process_requests(c)) {
      if (c.fd < 0) return;
      continue;  // drain input while the terminal response waits on output
    }
    if (c.in.size() > kMaxHeaderBytes + max_body_bytes_) {
      queue_response(c, 413, "text/plain",
                     "payload too large: http.max_body_bytes is " +
                         std::to_string(max_body_bytes_) + " bytes", true);
      flush_out(c);
      if (c.fd < 0) return;
    }
  }
}

bool HttpServer::process_requests(Conn& c) {
  while (!c.closing && !c.gone) {
    if (c.busy) return true;  // the handler still owes the previous answer
    // A complete request = headers + Content-Length body.
    const size_t head_end = c.in.find("\r\n\r\n");
    if (head_end == std::string::npos) {
      if (c.in.size() > kMaxHeaderBytes) {
        queue_response(c, 431, "text/plain", "headers too large", true);
        flush_out(c);
        return false;
      }
      return true;  // need more bytes
    }
    const size_t head_len = head_end + 4;
    if (head_len > kMaxHeaderBytes) {
      // The section cap applies to TERMINATED headers too — a single
      // 20 KiB header value is as oversized as an unterminated one.
      queue_response(c, 431, "text/plain", "headers too large", true);
      flush_out(c);
      return false;
    }

    // --- request line ---------------------------------------------------
    const size_t line_end = c.in.find("\r\n");
    const std::string_view req_line(c.in.data(), line_end);
    const size_t sp1 = req_line.find(' ');
    const size_t sp2 = req_line.find(' ', sp1 + 1);
    if (sp1 == std::string_view::npos || sp2 == std::string_view::npos) {
      queue_response(c, 400, "text/plain", "malformed request line", true);
      flush_out(c);
      return false;
    }
    HttpRequest req;
    req.method = std::string(req_line.substr(0, sp1));
    std::string_view target = req_line.substr(sp1 + 1, sp2 - sp1 - 1);
    const std::string_view version = req_line.substr(sp2 + 1);
    if (version != "HTTP/1.1") {
      queue_response(c, 400, "text/plain", "HTTP/1.1 only", true);
      flush_out(c);
      return false;
    }
    const size_t qmark = target.find('?');
    if (qmark == std::string_view::npos) {
      req.path = std::string(target);
    } else {
      req.path = std::string(target.substr(0, qmark));
      req.query = std::string(target.substr(qmark + 1));
    }

    // --- headers ----------------------------------------------------------
    bool has_te = false, has_len = false;
    size_t content_length = 0;
    size_t pos = line_end + 2;
    while (pos < head_end) {
      const size_t eol = c.in.find("\r\n", pos);
      if (eol == std::string::npos || eol > head_end) break;
      const std::string_view line(c.in.data() + pos, eol - pos);
      pos = eol + 2;
      if (line.empty()) continue;
      const size_t colon = line.find(':');
      if (colon == std::string_view::npos) {
        queue_response(c, 400, "text/plain", "malformed header line", true);
        flush_out(c);
        return false;
      }
      std::string key = to_lower(line.substr(0, colon));
      size_t vstart = colon + 1;
      while (vstart < line.size() && line[vstart] == ' ') ++vstart;
      std::string value(line.substr(vstart));
      if (key == "content-length") {
        std::string_view length(value);
        while (!length.empty() && (length.front() == ' ' || length.front() == '\t'))
          length.remove_prefix(1);
        while (!length.empty() && (length.back() == ' ' || length.back() == '\t'))
          length.remove_suffix(1);
        size_t parsed_length = 0;
        const auto [end, error] =
            std::from_chars(length.data(), length.data() + length.size(), parsed_length);
        if (error != std::errc{} || end != length.data() + length.size() ||
            (has_len && parsed_length != content_length)) {
          queue_response(c, 400, "text/plain", "invalid or conflicting Content-Length", true);
          flush_out(c);
          return false;
        }
        // Keep one canonical field so framing and handler-visible headers agree.
        if (has_len) continue;
        has_len = true;
        content_length = parsed_length;
        value = std::to_string(content_length);
      } else if (key == "transfer-encoding") {
        has_te = true;
      } else if (key == "connection") {
        const std::string cv = to_lower(value);
        c.keep_alive = cv != "close";
      }
      req.headers.emplace_back(std::move(key), std::move(value));
    }

    if (has_te) {
      // Chunked request bodies: refused loudly, not guessed.
      queue_response(c, 501, "text/plain",
                     "chunked request bodies unsupported — send Content-Length",
                     true);
      flush_out(c);
      return false;
    }
    if (req.method == "POST" && !has_len) {
      queue_response(c, 411, "text/plain", "Content-Length required", true);
      flush_out(c);
      return false;
    }
    if (content_length > max_body_bytes_) {
      queue_response(c, 413, "text/plain",
                     "payload too large: http.max_body_bytes is " +
                         std::to_string(max_body_bytes_) + " bytes", true);
      flush_out(c);
      return false;
    }
    if (c.in.size() < head_len + content_length) return true;  // need more

    req.body.assign(c.in, head_len, content_length);
    c.in.erase(0, head_len + content_length);  // keep-alive: next request

    // --- route ------------------------------------------------------------
    // One request in the handler's hands at a time: the writer and the
    // disconnect tag are per connection, so a second request dispatched
    // while the first is pending would orphan the first record (its tag
    // overwritten, its writer freed at the close without notice — the
    // fuzzer's third find, 2026-09-05). respond()/end_stream() clear busy.
    c.busy = true;
    const size_t response_start = c.out.size();
    try {
      handler_->handle(req, c.writer);
    } catch (const std::exception& e) {
      DGPP_LOG_ERROR("http: handler threw on {} {} — {}", req.method,
                     req.path, e.what());
      c.busy = false;
      if (c.fd < 0) return false;  // the handler killed the conn already
      // This handler has not flushed yet. Replace only its queued bytes;
      // an earlier pipelined reply may still be waiting on a writable socket.
      c.out.resize(response_start);
      c.stream = false;
      queue_response(c, 500, "text/plain", "internal server error", true);
    }
    if (c.fd < 0) return false;  // a disconnect raced the route
    flush_out(c);
    if (c.fd < 0) return false;  // EPIPE while flushing the response
  }
  return false;  // stop reading while the terminal response drains
}

void HttpServer::on_writable(Conn& c) { flush_out(c); }

void HttpServer::queue_response(Conn& c, int status,
                               std::string_view content_type,
                               const std::string& body, bool close_after) {
  std::string payload = body;
  if (status >= 400 && content_type == "text/plain") {
    payload = "{\"error\":{\"message\":";
    append_json_string(&payload, body);
    payload.append(status < 500 ? ",\"type\":\"invalid_request_error\""
                                : ",\"type\":\"server_error\"");
    payload.append(",\"param\":null,\"code\":");
    payload.append(status == 413 ? "\"request_too_large\"" : "null");
    payload.append("}}");
    content_type = "application/json";
  }
  char head[256];
  const int n = std::snprintf(
      head, sizeof(head),
      "HTTP/1.1 %d %s\r\nContent-Type: %.*s\r\nContent-Length: %zu\r\n"
      "Connection: close\r\n\r\n",
      status, status_text(status), static_cast<int>(content_type.size()),
      content_type.data(), payload.size());
  c.out.append(head, static_cast<size_t>(n));
  c.out.append(payload);
  if (close_after) c.closing = true;
}

bool HttpServer::flush_out(Conn& c) {
  while (c.flushed < c.out.size()) {
    const ssize_t put =
        ::send(c.fd, c.out.data() + c.flushed, c.out.size() - c.flushed,
               MSG_NOSIGNAL);
    if (put > 0) {
      c.flushed += static_cast<size_t>(put);
      if (c.stream) c.last_stream_write = std::chrono::steady_clock::now();
      continue;
    }
    if (put < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      // Arm EPOLLOUT so the remainder drains.
      epoll_event ev{};
      ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP;
      ev.data.fd = c.fd;
      (void)!::epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, c.fd, &ev);
      return true;
    }
    if (put < 0 && errno == EINTR) continue;
    c.gone = true;  // EPIPE & friends: the client left
    close_conn(c, /*notify_disconnect=*/true);
    return false;
  }
  // A successful send may have only made partial progress on earlier calls.
  // Close here so every flush path, including EPOLLOUT, drains the entire
  // queue before releasing the connection and notifying the handler.
  if (c.closing) {
    close_conn(c, /*notify_disconnect=*/false);
    return false;
  }
  // Fully drained: back to the plain read set.
  if (c.flushed >= c.out.size() && !c.out.empty()) {
    c.out.clear();
    c.flushed = 0;
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLRDHUP;
    ev.data.fd = c.fd;
    (void)!::epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, c.fd, &ev);
  }
  return true;
}

void HttpServer::close_conn(Conn& c, bool notify_disconnect) {
  if (c.fd < 0) return;
  const int fd = c.fd;
  (void)!::epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
  ::close(fd);
  c.fd = -1;
  c.gone = true;
  // Notify whenever the handler TAGGED the connection — streams obviously,
  // but also one-shot responses whose writer the handler still holds for
  // a later idle() flush — WHATEVER closed it: the client's goodbye, or
  // the server's own close after a 400/431 on a later pipelined request,
  // or the shutdown. The handler must then drop the writer: the Conn
  // (writer included) is freed at the pass's sweep. (The fuzzer's second
  // find, 2026-09-05 under ASan: a valid one-shot followed by garbage on
  // the same connection closed it without notice, and the service's
  // answer later wrote through the freed writer.)
  (void)notify_disconnect;
  if (c.stream_tag != 0) handler_->on_disconnect(c.stream_tag);
  // The node stays in the map (marked) until the pass's sweep — erasing
  // here would invalidate the caller's Conn&/iterator.
}

}  // namespace dgpp::serve
