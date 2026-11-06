#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dgpp::net {

// Deadline-correct POSIX TCP primitives for the M5 control plane.
//
// Policy: connect failures throw std::runtime_error (startup is exceptional
// and the caller wants the reason); read/write return false on timeout or
// peer close because the roster event loop treats those as protocol events,
// not errors. send() always uses MSG_NOSIGNAL so a dead peer can never raise
// SIGPIPE and kill the process.
class TcpConn {
 public:
  TcpConn() = default;
  TcpConn(TcpConn&& other) noexcept;
  TcpConn& operator=(TcpConn&& other) noexcept;
  TcpConn(const TcpConn&) = delete;
  TcpConn& operator=(const TcpConn&) = delete;
  ~TcpConn();

  // Connect with a hard timeout; throws on DNS/refused/timeout failure.
  // TCP_NODELAY is set: control-plane frames are small and latency-oriented.
  static TcpConn connect(const std::string& host, uint16_t port,
                         int timeout_ms);

  // SO_RCVTIMEO/SO_SNDTIMEO: read/write return false once exceeded.
  void set_io_deadline_ms(int ms);

  // Exact-length helpers; false on timeout, peer close, or error.
  bool write_all(const void* data, size_t len);
  bool write_all_bounded(const void* data, size_t len, int timeout_ms);
  bool read_exact(void* dst, size_t len);

  // One recv, no fill: n>0 bytes read, 0 = orderly peer close, -1 =
  // error or SO_RCVTIMEO expiry. Callers that wait_readable() first see
  // -1 only on genuine errors. The journal's line reader uses this so
  // a record costs one syscall, not one per byte; read_exact stays the
  // default for framed protocols.
  int read_some(void* dst, size_t cap);

  // poll(POLLIN) with deadline; true also when the peer closed (a subsequent
  // read_exact then returns false), so callers never block past the deadline.
  bool wait_readable(int timeout_ms);

  // Whether the peer has closed or reset the connection: a zero-timeout
  // poll and a one-byte MSG_PEEK — nothing is consumed, so another
  // thread's reader is undisturbed, and pending unread data reads as
  // "alive". The liveness probe of the fabric's death discipline
  // (fabric_serve.hpp): a rank's process death closes its sockets.
  bool peer_closed() const;
  // shutdown(SHUT_RDWR): unblocks a reader on another thread (its poll
  // sees EOF) and sends the peer a FIN — the tests' stand-in for a process
  // death without closing the descriptor under the reader.
  void shutdown_rw();

  void close();
  bool valid() const { return fd_ >= 0; }

 private:
  explicit TcpConn(int fd);

  int fd_ = -1;
  friend class TcpListener;
};

// Bind-and-listen helper. port 0 picks an ephemeral port (call port()).
class TcpListener {
 public:
  TcpListener() = default;
  TcpListener(TcpListener&& other) noexcept;
  TcpListener& operator=(TcpListener&& other) noexcept;
  TcpListener(const TcpListener&) = delete;
  TcpListener& operator=(const TcpListener&) = delete;
  ~TcpListener();

  static TcpListener bind(uint16_t port);

  uint16_t port() const { return port_; }

  // Blocks up to timeout_ms; nullopt on timeout. Accepted connections get
  // TCP_NODELAY but no I/O deadline — the caller sets one.
  // Returns an invalid conn only on unrecoverable accept errors (rare).
  TcpConn accept(int timeout_ms) const;

  void close();
  bool valid() const { return fd_ >= 0; }

 private:
  int fd_ = -1;
  uint16_t port_ = 0;
};

}  // namespace dgpp::net
