// M6 Stage 4: the HTTP/SSE server gate. A routing TestHandler stands in
// for the generation service; the test acts as a raw-socket client. No
// GPU, no model — this gate always runs, pinning:
//   * one-shot requests with keep-alive (a second request rides the same
//     connection);
//   * POST bodies by Content-Length (the handler sees the exact bytes);
//   * SSE: chunked framing, immediate events, and events drained across
//     idle() passes (the service's ring-drain shape);
//   * client disconnect mid-stream → on_disconnect(tag) — the scheduler
//     cancellation hook;
//   * the refusal ladder: malformed → 400 close, chunked request bodies
//     → 501, POST without length → 411, oversized headers → 431,
//     configurable body size → 413 before reading the body,
//     connection cap → 503 at the door;
//   * stop() returns the loop promptly.
#include "serve/http_server.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <limits>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "common/log.hpp"
#include "common/test.hpp"

namespace {

using dgpp::serve::HttpHandler;
using dgpp::serve::HttpRequest;
using dgpp::serve::HttpResponseWriter;
using dgpp::serve::HttpServer;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// --- a raw-socket client (the test never needs TcpConn's deadline
// discipline — it wants "read whatever arrived within N ms") ----------
class Client {
 public:
  explicit Client(uint16_t port, int receive_buffer = 0) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd_ >= 0, "client socket");
    if (receive_buffer > 0)
      require(::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &receive_buffer,
                          sizeof(receive_buffer)) == 0,
              "client receive buffer");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1,
            "client inet_pton");
    require(::connect(fd_, reinterpret_cast<sockaddr*>(&addr),
                      sizeof(addr)) == 0,
            "client connect");
    int yes = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  }
  ~Client() {
    if (fd_ >= 0) ::close(fd_);
  }
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  void send_all(std::string_view s) {
    size_t off = 0;
    while (off < s.size()) {
      const ssize_t put = ::send(fd_, s.data() + off, s.size() - off, 0);
      require(put > 0, "client send");
      off += static_cast<size_t>(put);
    }
  }
  // Everything that arrives within the timeout (EOF sets closed()).
  std::string read_available(int timeout_ms) {
    std::string out;
    int waited = 0;
    while (true) {
      pollfd p{fd_, POLLIN, 0};
      const int r = ::poll(&p, 1, 10);
      if (r < 0) {
        if (errno == EINTR) continue;
        break;
      }
      if (r == 0) {
        waited += 10;
        if (waited >= timeout_ms) break;
        continue;
      }
      char buf[4096];
      const ssize_t got = ::recv(fd_, buf, sizeof(buf), 0);
      if (got > 0) {
        out.append(buf, static_cast<size_t>(got));
        waited = 0;  // reset: responses may span segments
        continue;
      }
      if (got == 0) closed_ = true;
      break;
    }
    return out;
  }
  // The disconnect-test move: kill the socket without a goodbye byte.
  void hard_close() {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }
  bool closed() const { return closed_; }

 private:
  int fd_ = -1;
  bool closed_ = false;
};

// --- the TestHandler: routes the gate exercises -----------------------
class TestHandler : public HttpHandler {
 public:
  std::atomic<int> requests{0};
  std::atomic<int> disconnects{0};
  std::atomic<uint64_t> last_tag{0};

  // GET  /hello  → 200 JSON
  // POST /echo   → 200 body verbatim
  // GET  /gone   → 404
  // GET  /stream → SSE: 3 immediate events + end
  // GET  /drip   → SSE: 4 events from idle(), one per pass, + end
  // GET  /tagged → SSE with tag 42; only the client's close ends it
  void handle(const HttpRequest& req, HttpResponseWriter& w) override {
    ++requests;
    if (req.path == "/hello" && req.method == "GET") {
      w.respond(200, "application/json", "{\"ok\":true}");
    } else if (req.path == "/echo" && req.method == "POST") {
      w.respond(200, "text/plain", req.body);
    } else if (req.path == "/framing" && req.method == "POST") {
      size_t count = 0;
      std::string length;
      for (const auto& [key, value] : req.headers) {
        if (key == "content-length") {
          ++count;
          length = value;
        }
      }
      w.respond(200, "text/plain", std::to_string(count) + ":" + length);
    } else if (req.path == "/throw") {
      throw std::runtime_error("http-test internal diagnostic");
    } else if (req.path == "/stream" && req.method == "GET") {
      w.begin_stream();
      w.write_event("first");
      w.write_event("second");
      w.write_event("third");
      w.end_stream();
    } else if (req.path == "/comments" && req.method == "GET") {
      require(!w.write_comment("before"), "comments require an open stream");
      w.begin_stream();
      require(w.write_comment("keep-alive"), "comment queued");
      bool rejected = false;
      try {
        w.write_comment("unsafe\ndata: injected");
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "a comment cannot inject an SSE event");
      w.write_event("[DONE]");
      w.end_stream();
      require(!w.write_comment("after"), "no comments after termination");
      w.ping_if_idle(1);
    } else if (req.path == "/drip" && req.method == "GET") {
      w.begin_stream();
      drip_.push_back(&w);  // drained from idle(), like the service
    } else if (req.path == "/tagged" && req.method == "GET") {
      // Silent stream: only the client's close ends it (the drip list
      // would end_stream() after 4 events, defusing the disconnect).
      w.begin_stream();
      w.set_stream_tag(42);
    } else {
      w.respond(404, "application/json", "{\"error\":\"no route\"}");
    }
  }

  void idle() override {
    // One event per pass per drip stream; the 4th event ends it.
    for (auto* w : drip_) {
      if (w->stream_open()) {
        w->write_event("drip");
        if (++drip_sent_ >= 4) {
          w->end_stream();
          drip_sent_ = 0;
        }
      }
    }
  }

  void on_disconnect(uint64_t tag) override {
    ++disconnects;
    last_tag = tag;
    drip_.clear();
  }

 private:
  std::vector<HttpResponseWriter*> drip_;  // epoll-thread only
  int drip_sent_ = 0;
};

// Server + serve thread; stops/joins in the destructor so a failed
// require still unwinds cleanly.
struct ServerHandle {
  TestHandler handler;
  HttpServer server;
  std::thread loop;

  explicit ServerHandle(int max_connections = 64,
                        int64_t max_body_bytes = dgpp::serve::kDefaultHttpMaxBodyBytes)
      : server(0, &handler, max_connections, "127.0.0.1", max_body_bytes),
        loop([this] { server.serve(); }) {}
  ~ServerHandle() {
    server.stop();
    if (loop.joinable()) loop.join();  // the stop test joins first
  }
  ServerHandle(const ServerHandle&) = delete;
  ServerHandle& operator=(const ServerHandle&) = delete;

  uint16_t port() const { return server.port(); }
};

const char* kGetHello = "GET /hello HTTP/1.1\r\nHost: t\r\n\r\n";

DGPP_TEST(http_default_bind_does_not_accept_other_loopback_addresses) {
  ServerHandle sh;
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "socket");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(sh.port());
  ::inet_pton(AF_INET, "127.0.0.2", &address.sin_addr);
  const int result = ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
  ::close(fd);
  require(result != 0, "default listener must bind only 127.0.0.1, not INADDR_ANY");
}

DGPP_TEST(http_bind_address_is_configurable_and_validated) {
  TestHandler handler;
  HttpServer server(0, &handler, 64, "127.0.0.2");
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "socket");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(server.port());
  ::inet_pton(AF_INET, "127.0.0.2", &address.sin_addr);
  const int result = ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
  ::close(fd);
  require(result == 0, "explicit bind address accepts connections");
  bool refused = false;
  try { HttpServer invalid(0, &handler, 64, "localhost"); }
  catch (const std::exception&) { refused = true; }
  require(refused, "non-IPv4 bind address rejected");
}

DGPP_TEST(http_oneShot_keepAliveSecondRequestSameConnection) {
  // GIVEN a running server,
  ServerHandle sh;

  // WHEN two requests ride one connection,
  Client c(sh.port());
  c.send_all(kGetHello);
  const std::string first = c.read_available(500);
  c.send_all(kGetHello);
  const std::string second = c.read_available(500);

  // THEN both answer 200 with the exact body, and the connection stays
  // open for the second (no close byte between them).
  require(first.find("200 OK") != std::string::npos &&
              first.find("{\"ok\":true}") != std::string::npos,
          "first response: " + first);
  require(second.find("200 OK") != std::string::npos &&
              second.find("{\"ok\":true}") != std::string::npos,
          "second response (keep-alive): " + second);
}

DGPP_TEST(http_postBody_contentLengthDeliveredVerbatim) {
  // GIVEN a running server,
  ServerHandle sh;

  // WHEN a POST carries a JSON body by Content-Length,
  Client c(sh.port());
  const std::string body = "{\"text\":\"hello world\",\"n\":7}";
  c.send_all("POST /echo HTTP/1.1\r\nHost: t\r\nContent-Type: "
             "application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);

  // THEN the handler's echo returns the exact bytes (the parser did not
  // clip or buffer-shift the body).
  const std::string resp = c.read_available(500);
  require(resp.find("200 OK") != std::string::npos,
          "echo status: " + resp);
  require(resp.find(body) != std::string::npos, "echo body: " + resp);
}

DGPP_TEST(http_content_length_accepts_decimal_whitespace_and_equal_repeated_fields) {
  ServerHandle sh;
  Client c(sh.port());
  struct Case {
    const char* headers;
    const char* body;
  };
  const Case cases[] = {
      {"Content-Length: 4\r\n", "test"},
      {"Content-Length: \t0004 \t\r\n", "test"},
      {"Content-Length: 04\r\ncontent-length:\t4\r\n", "test"},
      {"Content-Length: 0\r\nContent-Length: 00\r\n", ""},
  };
  for (const auto& test : cases) {
    c.send_all(std::string("POST /echo HTTP/1.1\r\nHost: t\r\n") + test.headers + "\r\n" +
               test.body);
    const auto response = c.read_available(100);
    require(response.starts_with("HTTP/1.1 200 OK\r\n"), "valid decimal length accepted");
    require(response.substr(response.find("\r\n\r\n") + 4) == test.body,
            "equivalent length fields preserve the exact body on keep-alive");
  }
  require(sh.handler.requests.load() == 4, "all valid requests reach the handler");
}

DGPP_TEST(http_content_length_handler_sees_one_canonical_field) {
  ServerHandle sh;
  Client c(sh.port());
  struct Case {
    const char* headers;
    const char* body;
    const char* expected;
  };
  const Case cases[] = {
      {"Content-Length: \t0004 \t\r\n", "test", "1:4"},
      {"Content-Length: 04\r\ncontent-length:\t4\r\n", "test", "1:4"},
      {"Content-Length: 0\r\nContent-Length: 00\r\n", "", "1:0"},
  };
  for (const auto& test : cases) {
    c.send_all(std::string("POST /framing HTTP/1.1\r\nHost: t\r\n") + test.headers + "\r\n" +
               test.body);
    const auto response = c.read_available(500);
    require(response.starts_with("HTTP/1.1 200 OK\r\n"), "framing request succeeds");
    require(response.substr(response.find("\r\n\r\n") + 4) == test.expected,
            "handler sees exactly one canonical Content-Length");
  }
  require(sh.handler.requests.load() == 3, "all framing observations reach the handler");
}

DGPP_TEST(http_invalid_or_conflicting_content_length_400_closes_without_dispatch) {
  ServerHandle sh(64, 1024);
  const auto rejected = [&](const std::string& headers) {
    Client c(sh.port());
    c.send_all("POST /echo HTTP/1.1\r\nHost: t\r\n" + headers + "\r\ntest");
    const auto response = c.read_available(500);
    require(response.starts_with("HTTP/1.1 400 Bad Request\r\n"), "invalid length is a 400");
    require(
        response.find("\"message\":\"invalid or conflicting Content-Length\"") != std::string::npos,
        "refusal names the invalid header");
    require(response.find("Content-Type: application/json\r\n") != std::string::npos,
            "transport refusal retains the JSON error envelope");
    require(response.find("Connection: close\r\n") != std::string::npos && c.closed(),
            "invalid framing closes the connection");
  };
  for (const auto& value :
       std::vector<std::string>{"", " \t", "+4", "-0", "-1", "4x", "4.0", "0x4", "4 0", "4, 4",
                                "4, 5", std::string(128, '9'), std::string("4\0x", 3)})
    rejected("Content-Length: " + value + "\r\n");
  rejected("Content-Length: 3\r\nContent-Length: 4\r\n");
  rejected("Content-Length: 4\r\nContent-Length: 3\r\n");
  rejected("Content-Length: 4\r\nContent-Length: x\r\n");
  require(sh.handler.requests.load() == 0, "invalid framing never reaches the handler");
}

DGPP_TEST(http_handler_exception_returns_generic_500_and_closes) {
  ServerHandle sh;
  Client c(sh.port());
  c.send_all("GET /throw HTTP/1.1\r\nHost: t\r\n\r\n");
  const auto response = c.read_available(500);
  require(response.starts_with("HTTP/1.1 500 Internal Server Error\r\n"),
          "handler exception is a 500");
  require(response.find("Content-Type: application/json\r\n") != std::string::npos &&
              response.find("\"message\":\"internal server error\"") != std::string::npos,
          "unexpected error uses a generic JSON message");
  require(response.find("http-test internal diagnostic") == std::string::npos,
          "internal exception details do not reach the client");
  require(response.find("Connection: close\r\n") != std::string::npos && c.closed(),
          "unexpected error closes the connection");
}

DGPP_TEST(http_content_length_validates_get_and_size_t_boundaries) {
  ServerHandle sh(64, 1024);
  const auto maximum = std::to_string(std::numeric_limits<size_t>::max());
  for (const auto& value : std::vector<std::string>{"x", "-0", "+0", maximum + "0"}) {
    Client c(sh.port());
    c.send_all("GET /hello HTTP/1.1\r\nContent-Length: " + value + "\r\n\r\n");
    require(c.read_available(500).starts_with("HTTP/1.1 400 ") && c.closed(),
            "invalid GET length is rejected before dispatch");
  }
  {
    Client c(sh.port());
    c.send_all("GET /hello HTTP/1.1\r\nContent-Length: " + maximum + "\r\n\r\n");
    require(c.read_available(500).starts_with("HTTP/1.1 413 ") && c.closed(),
            "largest representable length reaches the body cap check without overflow");
  }
  {
    Client c(sh.port());
    c.send_all("GET /hello HTTP/1.1\r\nContent-Length: " + std::string(5000, '0') + "\r\n\r\n");
    require(c.read_available(100).starts_with("HTTP/1.1 200 "),
            "many leading zeros still represent zero");
  }
  require(sh.handler.requests.load() == 1, "only the valid zero-length GET is dispatched");
}

DGPP_TEST(http_refusals_drain_prior_response_before_closing_under_backpressure) {
  struct Case {
    std::string request;
    const char* status;
    int dispatched;
    bool late_input = false;
  };
  const Case cases[] = {
      {"GET / HTTP/1.1\r\nContent-Length: +0\r\n\r\n", "400 Bad Request", 1},
      {"GET / HTTP/1.1\r\nContent-Length: +0\r\n\r\n", "400 Bad Request", 1, true},
      {"GET / HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n", "400 Bad Request", 1},
      {"bad-request\r\n\r\n", "400 Bad Request", 1},
      {"GET / HTTP/1.0\r\n\r\n", "400 Bad Request", 1},
      {"GET / HTTP/1.1\r\nBad-Header\r\n\r\n", "400 Bad Request", 1},
      {"POST / HTTP/1.1\r\n\r\n", "411 Length Required", 1},
      {"POST / HTTP/1.1\r\nContent-Length: 1025\r\n\r\n", "413 Content Too Large", 1},
      {"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", "501 Not Implemented", 1},
      {"GET / HTTP/1.1\r\nX-Large: " + std::string(17 * 1024, 'x') + "\r\n\r\n",
       "431 Request Header Fields Too Large", 1},
      {"GET / HTTP/1.1\r\nX-Large: " + std::string(17 * 1024, 'x'),
       "431 Request Header Fields Too Large", 1},
      {"GET /throw HTTP/1.1\r\n\r\n", "500 Internal Server Error", 2},
  };
  for (const auto& test : cases) {
    struct Handler : HttpHandler {
      std::atomic<bool> processed{false};
      std::atomic<int> requests{0};
      std::atomic<int> disconnects{0};
      void handle(const HttpRequest& req, HttpResponseWriter& w) override {
        ++requests;
        w.set_stream_tag(42);
        if (req.path == "/large") {
          w.respond(200, "text/plain", std::string(16 * 1024 * 1024, 'z'));
        } else if (req.path == "/throw") {
          w.respond(200, "text/plain", "discard this unfinished response");
          throw std::runtime_error("backpressure test diagnostic");
        } else {
          w.respond(200, "text/plain", "unexpected dispatch");
        }
      }
      void idle() override {
        if (requests.load() > 0) processed = true;
      }
      void on_disconnect(uint64_t) override { ++disconnects; }
    } handler;
    HttpServer server(0, &handler, 8, "127.0.0.1", 1024);
    Client client(server.port(), 64 * 1024);
    // Queue both requests before the server runs, then leave the receive
    // window full until it has parsed the refusal behind the large reply.
    client.send_all("GET /large HTTP/1.1\r\n\r\n" + test.request);
    std::thread loop([&] { server.serve(); });
    struct Join {
      HttpServer& server;
      std::thread& loop;
      ~Join() {
        server.stop();
        if (loop.joinable()) loop.join();
      }
    } join{server, loop};
    for (int i = 0; i < 1000 && !handler.processed.load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(handler.processed.load(), "server processed pipelined requests");
    if (test.late_input) client.send_all("GET /late HTTP/1.1\r\n\r\n");
    const auto response = client.read_available(1000);
    server.stop();
    loop.join();
    const auto first_body = response.find("\r\n\r\n") + 4;
    require(response.starts_with("HTTP/1.1 200 OK\r\n") &&
                response.substr(first_body, 16 * 1024 * 1024) ==
                    std::string(16 * 1024 * 1024, 'z'),
            std::string(test.status) + ": earlier response must arrive intact");
    const auto refusal = response.substr(first_body + 16 * 1024 * 1024);
    const auto length_header = refusal.find("Content-Length: ");
    const auto refusal_body = refusal.find("\r\n\r\n");
    require(refusal.starts_with(std::string("HTTP/1.1 ") + test.status + "\r\n") &&
                refusal.find("Content-Type: application/json\r\n") != std::string::npos &&
                refusal.ends_with("}}") && client.closed(),
            std::string(test.status) + ": complete JSON refusal precedes EOF");
    require(length_header != std::string::npos && refusal_body != std::string::npos &&
                std::stoull(refusal.substr(length_header + 16)) == refusal.size() - refusal_body - 4,
            "refusal body matches its declared length with no additional responses");
    require(refusal.find("backpressure test diagnostic") == std::string::npos &&
                refusal.find("discard this unfinished response") == std::string::npos,
            "only the failed handler's pending response is replaced");
    require(handler.requests.load() == test.dispatched && handler.disconnects.load() == 1,
            "refused requests are not dispatched and the tagged connection closes once");
  }
}

DGPP_TEST(http_default_body_limit_accepts_large_prefill_payloads) {
  ServerHandle sh;
  Client c(sh.port());
  const std::string body = "{\"document\":\"" + std::string(5 * 1024 * 1024, 'x') + "\"}";
  c.send_all("POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string response = c.read_available(500);
  require(response.starts_with("HTTP/1.1 200 OK\r\n"), "payload above the former 4 MiB cap accepted");
  require(response.substr(response.find("\r\n\r\n") + 4) == body,
          "large serialized document reaches the handler intact");
}

DGPP_TEST(http_body_limit_applies_per_request_at_the_configured_boundary) {
  constexpr int64_t limit = 32768;
  ServerHandle sh(64, limit);
  Client c(sh.port());
  for (const size_t size : {limit - 1, limit}) {
    const std::string body(size, 'x');
    c.send_all("POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: " +
               std::to_string(size) + "\r\n\r\n");
    c.send_all(std::string_view(body).substr(0, size - 1));
    require(c.read_available(50).empty(), "partial body waits for its final byte");
    c.send_all(std::string_view(body).substr(size - 1));
    const std::string response = c.read_available(100);
    require(response.starts_with("HTTP/1.1 200 OK\r\n"), "body at or below configured cap accepted");
    require(response.substr(response.find("\r\n\r\n") + 4) == body,
            "fragmented body reaches handler intact on reused connection");
  }
  c.send_all("POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: " +
             std::to_string(limit + 1) + "\r\n\r\n");
  const std::string response = c.read_available(500);
  require(response.starts_with("HTTP/1.1 413 Content Too Large\r\n"),
          "one byte over cap rejected from headers alone");
  require(response.find("http.max_body_bytes is 32768 bytes") != std::string::npos,
          "refusal identifies the configured byte cap");
  require(response.find("Content-Type: application/json\r\n") != std::string::npos &&
              response.find("\"error\":{\"message\":") != std::string::npos &&
              response.find("\"code\":\"request_too_large\"") != std::string::npos,
          "transport refusal uses the OpenAI error envelope");
  require(response.find("Connection: close\r\n") != std::string::npos, "oversized request closes connection");
  require(sh.handler.requests.load() == 2, "oversized request never reaches handler");
}

DGPP_TEST(http_body_limit_can_exceed_default) {
  const int64_t length = dgpp::serve::kDefaultHttpMaxBodyBytes + 1;
  ServerHandle normal;
  ServerHandle larger(64, 2 * dgpp::serve::kDefaultHttpMaxBodyBytes);
  const std::string headers = "POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: " +
                              std::to_string(length) + "\r\n\r\n";
  Client a(normal.port());
  a.send_all(headers);
  require(a.read_available(500).starts_with("HTTP/1.1 413 Content Too Large\r\n"),
          "default still bounds uploads");
  Client b(larger.port());
  b.send_all(headers);
  require(b.read_available(100).empty(), "larger configured cap permits a body above the default");
  b.hard_close();
  require(normal.handler.requests.load() == 0 && larger.handler.requests.load() == 0,
          "headers alone do not invoke the handler");
}

DGPP_TEST(http_body_limit_must_be_positive) {
  TestHandler handler;
  for (const int64_t limit : {0, -1}) {
    bool refused = false;
    try { HttpServer invalid(0, &handler, 64, "127.0.0.1", limit); }
    catch (const std::invalid_argument&) { refused = true; }
    require(refused, "nonpositive body cap rejected before opening listener");
  }
}

DGPP_TEST(http_malformedRequestLine_400AndClose) {
  // GIVEN a running server,
  ServerHandle sh;

  // WHEN garbage arrives,
  Client c(sh.port());
  c.send_all("not-a-request-line\r\n\r\n");

  // THEN the server answers 400 and closes (the read hits EOF).
  const std::string resp = c.read_available(500);
  require(resp.find("400") != std::string::npos, "400 on garbage: " + resp);
  const std::string tail = c.read_available(500);
  require(resp.find("400") != std::string::npos && tail.empty(),
          "connection closed after the 400");
}

DGPP_TEST(http_chunkedRequestBody_refused501) {
  // GIVEN a running server,
  ServerHandle sh;

  // WHEN a request declares Transfer-Encoding: chunked (we only accept
  // Content-Length bodies),
  Client c(sh.port());
  c.send_all("POST /echo HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: "
             "chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n");

  // THEN it is refused loudly with 501 — never guessed at.
  const std::string resp = c.read_available(500);
  require(resp.find("501") != std::string::npos,
          "chunked request bodies must be refused: " + resp);
}

DGPP_TEST(http_postWithoutContentLength_411) {
  // GIVEN a running server,
  ServerHandle sh;

  // WHEN a POST arrives with no body length,
  Client c(sh.port());
  c.send_all("POST /echo HTTP/1.1\r\nHost: t\r\n\r\n");

  // THEN the server demands one (411).
  const std::string resp = c.read_available(500);
  require(resp.find("411") != std::string::npos,
          "length required: " + resp);
}

DGPP_TEST(http_sseStream_chunkedEventsInOrderWithTerminal) {
  // GIVEN a running server,
  ServerHandle sh;

  // WHEN an immediate-event stream is requested,
  Client c(sh.port());
  c.send_all("GET /stream HTTP/1.1\r\nHost: t\r\n\r\n");

  // THEN the response is SSE over chunked framing: the stream headers,
  // three data events in order, and the terminal chunk.
  const std::string resp = c.read_available(500);
  require(resp.find("200 OK") != std::string::npos &&
              resp.find("text/event-stream") != std::string::npos &&
              resp.find("chunked") != std::string::npos,
          "stream headers: " + resp.substr(0, 200));
  const size_t first = resp.find("data: first");
  const size_t second = resp.find("data: second");
  const size_t third = resp.find("data: third");
  const size_t terminal = resp.find("0\r\n\r\n");
  require(first != std::string::npos && second != std::string::npos &&
              third != std::string::npos,
          "all three events present");
  require(first < second && second < third, "events in order");
  require(terminal != std::string::npos && terminal > third,
          "terminal chunk after the last event");
}

DGPP_TEST(http_sseComments_chunkedFramingAndStreamBoundaries) {
  ServerHandle sh;
  Client client(sh.port());
  client.send_all("GET /comments HTTP/1.1\r\nHost: t\r\n\r\n");
  const auto response = client.read_available(100);
  const auto body = response.substr(response.find("\r\n\r\n") + 4);
  require(body == "e\r\n: keep-alive\n\n\r\ne\r\ndata: [DONE]\n\n\r\n0\r\n\r\n",
          "comment and event have distinct SSE framing inside valid HTTP chunks: " + body);
}

DGPP_TEST(http_sseIdleDrain_eventsArriveAcrossLoopPasses) {
  // GIVEN a running server whose drip stream emits one event per idle
  // pass,
  ServerHandle sh;

  // WHEN the stream is requested,
  Client c(sh.port());
  c.send_all("GET /drip HTTP/1.1\r\nHost: t\r\n\r\n");

  // THEN four drip events arrive (each on its own loop pass — this is
  // the service's ring-drain shape) followed by the terminal chunk.
  const std::string resp = c.read_available(2000);
  int drips = 0;
  for (size_t pos = resp.find("data: drip"); pos != std::string::npos;
       pos = resp.find("data: drip", pos + 1))
    ++drips;
  require(drips == 4, "exactly four drip events (got " +
                         std::to_string(drips) + ")");
  require(resp.find("0\r\n\r\n") != std::string::npos,
          "the stream terminated");
}

DGPP_TEST(http_clientDisconnectMidStream_firesOnDisconnectWithTag) {
  // GIVEN a running server with a tagged stream,
  ServerHandle sh;
  Client c(sh.port());
  c.send_all("GET /tagged HTTP/1.1\r\nHost: t\r\n\r\n");
  require(!c.read_available(300).empty(), "stream headers arrived");

  // WHEN the client vanishes mid-stream,
  c.hard_close();

  // THEN on_disconnect fires with the tag (the service's cancellation
  // hook) within the loop's cadence.
  for (int waited = 0; sh.handler.disconnects.load() == 0 && waited < 2000;
       waited += 10)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  require(sh.handler.disconnects.load() == 1,
          "on_disconnect fired exactly once");
  require(sh.handler.last_tag.load() == 42, "the stream tag reached us");
}

DGPP_TEST(http_oversizedHeaders_431) {
  // GIVEN a running server,
  ServerHandle sh;

  // WHEN headers exceed the 16 KiB cap with no end in sight,
  Client c(sh.port());
  std::string huge = "GET /hello HTTP/1.1\r\nHost: t\r\nX-Big: ";
  huge.append(20 * 1024, 'a');
  huge.append("\r\n\r\n");
  c.send_all(huge);

  // THEN the server refuses with 431 and closes.
  const std::string resp = c.read_available(500);
  require(resp.find("431") != std::string::npos, "431 on huge headers: " +
                                                     resp.substr(0, 80));
}

DGPP_TEST(http_connectionCap_shedsAtTheDoorWith503) {
  // GIVEN a server that admits exactly one connection,
  ServerHandle sh(/*max_connections=*/1);

  // WHEN a second client arrives while the first holds its connection,
  Client holder(sh.port());
  holder.send_all(kGetHello);
  require(!holder.read_available(500).empty(), "holder served");

  Client extra(sh.port());

  // THEN the extra connection is shed at the door with 503 — no state
  // allocated, no request parsed.
  const std::string resp = extra.read_available(500);
  require(resp.find("503") != std::string::npos, "503 at the door: " + resp);
}

DGPP_TEST(http_stop_serveLoopReturnsPromptly) {
  // GIVEN a running server,
  ServerHandle sh;

  // WHEN stop() is called from this (non-HTTP) thread,
  const auto t0 = std::chrono::steady_clock::now();
  sh.server.stop();
  sh.loop.join();
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

  // THEN the loop exits within one epoll timeout's grace (25ms bound,
  // generous allowance for scheduling). The destructor's own stop+join
  // is a no-op thanks to joinable().
  require(ms < 500.0, "stop returned promptly (took " +
                         std::to_string(ms) + "ms)");
}

}  // namespace

DGPP_TEST(http_wake_pending_concurrent_and_late_notifications) {
  struct Handler : HttpHandler {
    dgpp::serve::HttpWakeup wake;
    std::mutex mutex;
    std::condition_variable changed;
    int passes = 0;
    void handle(const HttpRequest&, HttpResponseWriter&) override {}
    void set_wakeup(dgpp::serve::HttpWakeup value) override { wake = value; }
    void idle() override {
      std::lock_guard lock(mutex);
      ++passes;
      changed.notify_all();
    }
    bool wait_for_pass(int count) {
      std::unique_lock lock(mutex);
      return changed.wait_for(lock, std::chrono::milliseconds(500),
                              [&] { return passes >= count; });
    }
  } handler;
  {
    HttpServer polling(0, &handler, 8);
    require(!handler.wake && !polling.wakeup(), "default polling allocates no notification handle");
  }
  dgpp::serve::HttpWakeup late;
  {
    HttpServer server(0, &handler, 8, "127.0.0.1", dgpp::serve::kDefaultHttpMaxBodyBytes, true,
                      5000);
    late = server.wakeup();
    for (int i = 0; i < 100; ++i) handler.wake();  // before epoll_wait starts
    std::thread loop([&] { server.serve(); });
    struct Join {
      HttpServer& server;
      std::thread& loop;
      ~Join() {
        server.stop();
        loop.join();
      }
    } join{server, loop};
    require(handler.wait_for_pass(1), "pending wake interrupts a five-second poll");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
      std::lock_guard lock(handler.mutex);
      require(handler.passes == 1, "a burst coalesces and the counter is drained without spinning");
    }
    std::vector<std::thread> producers;
    for (int i = 0; i < 4; ++i)
      producers.emplace_back([wake = handler.wake] {
        for (int j = 0; j < 100; ++j) wake();
      });
    for (auto& producer : producers) producer.join();
    require(handler.wait_for_pass(2), "concurrent producers wake a waiting loop");
  }
  // Copies own the eventfd independently of server lifetime. A delayed job
  // can notify safely even while unrelated sockets are allocated and closed.
  for (int i = 0; i < 100; ++i) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    late();
    ::close(fd);
  }
}

DGPP_TEST(http_wake_delivers_producer_output_and_terminal_event) {
  struct Handler : HttpHandler {
    dgpp::serve::HttpWakeup wake;
    HttpResponseWriter* writer = nullptr;
    std::mutex mutex;
    std::vector<std::string> pending;
    bool done = false;
    void set_wakeup(dgpp::serve::HttpWakeup value) override { wake = value; }
    void handle(const HttpRequest&, HttpResponseWriter& w) override {
      w.begin_stream();
      writer = &w;
    }
    void idle() override {
      std::lock_guard lock(mutex);
      if (!writer) return;
      for (const auto& event : pending) writer->write_event(event);
      pending.clear();
      if (done) {
        writer->write_event("[DONE]");
        writer->end_stream();
        writer = nullptr;
      }
    }
  } handler;
  HttpServer server(0, &handler, 8, "127.0.0.1", dgpp::serve::kDefaultHttpMaxBodyBytes, true, 5000);
  std::thread loop([&] { server.serve(); });
  struct Join {
    HttpServer& server;
    std::thread& loop;
    ~Join() {
      server.stop();
      loop.join();
    }
  } join{server, loop};
  Client client(server.port());
  client.send_all("GET / HTTP/1.1\r\nHost: t\r\n\r\n");
  require(client.read_available(50).find("200 OK") != std::string::npos, "headers");
  {
    std::lock_guard lock(handler.mutex);
    handler.pending = {"first", "second"};
    handler.done = true;
  }
  handler.wake();
  const auto output = client.read_available(500);
  require(output.find("data: first\n\n") != std::string::npos &&
              output.find("data: second\n\n") != std::string::npos &&
              output.find("data: first\n\n") < output.find("data: second\n\n") &&
              output.find("data: [DONE]\n\n") != std::string::npos && output.ends_with("0\r\n\r\n"),
          "wake drains ordered events and terminates the stream");
}

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  return dgpp::test::run_all();
}
