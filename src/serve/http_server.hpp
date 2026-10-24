#pragma once
// HTTP/1.1 and SSE server using a single epoll loop. Socket ownership
// stays on the HTTP thread; handlers queue model work through the service.
//
// SHAPE (one thread owns every socket):
//   * A single epoll loop accepts, parses, routes, and writes. Route
//     handlers run INLINE on that thread and must not block — admission
//     to the generation service is a queue append, never a wait.
//   * SSE streams: the handler calls begin_stream() and keeps the
//     writer; token events arrive later on the ENGINE thread, so the
//     service buffers them in its own per-request rings and drains them
//     from idle(), once per loop pass. By default a quiet loop polls every
//     25 ms; opt-in output notifications wake it sooner. Sockets are only
//     ever written by the epoll thread.
//   * Client disconnects during a stream surface as on_disconnect(tag)
//     — the service maps that onto scheduler cancellation.
//
// SUPPORTED (deliberately small, refused loudly everywhere else):
// HTTP/1.1 keep-alive; request bodies by Content-Length only (chunked
// REQUEST bodies → 501; curl and every OpenAI client send lengths); SSE
// responses via chunked transfer encoding. Limits: 16 KiB of headers,
// a configurable body cap (default 256 MiB) and connection cap — beyond them,
// 431/413/503 with the connection closed. Malformed input → 400 close.
// Fuzzing hardening is M9; the parser is defensive anyway (it runs in
// the same process as the engine).
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "serve/http_limits.hpp"

namespace dgpp::serve {

struct HttpRequest {
  std::string method;
  std::string path;    // path only, no query string
  std::string query;   // raw query string ("" when absent)
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;  // keys lowercased

  // Case-insensitive header lookup (nullptr = absent).
  const std::string* header(std::string_view name) const;
};

// Internal per-connection state (defined in http_server.cpp).
struct Conn;

// The response interface handed to route handlers. Epoll-thread only (the
// generation service's engine thread reaches streams through idle(),
// never through this class). Each connection owns one writer for its
// lifetime; a handler may keep the pointer beyond handle() for stream
// writes from idle(), but MUST drop it when on_disconnect fires (the
// connection — and its writer — is freed at the loop pass's sweep).
class HttpResponseWriter {
 public:
  HttpResponseWriter() = default;  // detached; the server attaches per-conn

  // One-shot response. The connection stays open when keep-alive is on.
  void respond(int status, std::string_view content_type, std::string body);

  // --- SSE streaming ---------------------------------------------------
  // Sends the stream headers and marks the connection chunked.
  void begin_stream(std::string_view content_type = "text/event-stream");
  // One `data: <data>\n\n` event (no trailing newline in `data`).
  // Returns false once the client is gone (later events are dropped;
  // on_disconnect has already fired).
  bool write_event(std::string_view data);
  // One SSE comment (no CR/LF in comment), invisible to event consumers.
  bool write_comment(std::string_view comment);
  // Called by the HTTP idle pass after draining events. The silence clock
  // belongs to the connection, including when multiple choices share it.
  // A non-positive interval disables pings; pending output suppresses them.
  void ping_if_idle(int interval_s);
  // Ends the stream (the terminal chunk) and keeps the connection open
  // under keep-alive.
  void end_stream();

  // Stable correlation for on_disconnect: the handler sets this when it
  // opens a stream (e.g. its request id).
  void set_stream_tag(uint64_t tag);
  uint64_t stream_tag() const;

  bool stream_open() const;
  bool client_gone() const;

 private:
  friend class HttpServer;
  void attach(Conn* conn) { conn_ = conn; }
  Conn* conn_ = nullptr;  // null or closed conn: writes become no-ops
};

// Copyable producer-side notification. Its descriptor stays alive until the
// last copy is released, so late background work never writes a recycled fd
// after the server stops or is destroyed. A default handle is a no-op.
class HttpWakeup {
 public:
  HttpWakeup() = default;
  void operator()() const noexcept;
  explicit operator bool() const { return static_cast<bool>(state_); }

 private:
  friend class HttpServer;
  struct State;
  std::shared_ptr<State> state_;
};

class HttpHandler {
 public:
  virtual ~HttpHandler() = default;
  // Route the request. May throw — the server answers 500 and closes;
  // the engine thread never sees it.
  virtual void handle(const HttpRequest& req, HttpResponseWriter& w) = 0;
  // Installed at server construction, before producer threads start. Keep a
  // copy in background jobs; publish their result before calling it.
  virtual void set_wakeup(HttpWakeup wakeup) { (void)wakeup; }
  // Background request preparation always notifies, independently of the
  // policy for flushing generated tokens.
  virtual bool needs_work_wakeup() const { return false; }
  virtual void set_work_wakeup(HttpWakeup wakeup) { (void)wakeup; }
  // Once per loop pass: drain SSE rings, re-check liveness.
  virtual void idle() {}
  // A TAGGED connection closed — mid-stream, or before a one-shot
  // response finished flushing. `tag` is the value set via
  // set_stream_tag (0 = the handler never tagged it, no notification).
  // The handler must drop the writer: the connection (writer included)
  // is freed at the loop pass's sweep.
  virtual void on_disconnect(uint64_t tag) { (void)tag; }
};

class HttpServer {
 public:
  // `max_connections` beyond which accept answers 503 + close.
  // `max_body_bytes` is a positive byte ceiling, checked from Content-Length;
  // buffers grow with received data, not with the configured ceiling.
  // poll_timeout_ms must be positive; embedders and host tests may override
  // the normal 25 ms maintenance poll.
  HttpServer(uint16_t port, HttpHandler* handler, int max_connections,
             const std::string& bind_host = "127.0.0.1",
             int64_t max_body_bytes = kDefaultHttpMaxBodyBytes, bool wake_on_output = false,
             int poll_timeout_ms = 25);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // The bound port — call after construction (port 0 picks an ephemeral
  // port; this reports what the kernel chose).
  uint16_t port() const { return port_; }

  // The blocking epoll loop (run on the HTTP thread). Returns after
  // stop() (or an unrecoverable setup error — construction throws
  // first for those).
  void serve();

  // Thread-safe; wakes the loop in notification mode, otherwise the loop
  // notices within one timeout pass.
  void stop();

  HttpWakeup wakeup() const { return wakeup_; }

 private:
  void accept_new();
  void on_readable(Conn& c);
  void on_writable(Conn& c);
  void close_conn(Conn& c, bool notify_disconnect);
  bool flush_out(Conn& c);
  // Parses every complete request from c.in; false when input must stop
  // (the connection is closed or its terminal response is still draining).
  bool process_requests(Conn& c);
  void queue_response(Conn& c, int status, std::string_view content_type,
                      const std::string& body, bool close_after);

  uint16_t port_;
  HttpHandler* handler_;
  int max_connections_;
  size_t max_body_bytes_;
  int poll_timeout_ms_;
  HttpWakeup wakeup_;
  int listen_fd_ = -1;
  int epoll_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::map<int, std::unique_ptr<Conn>> conns_;
};

}  // namespace dgpp::serve
