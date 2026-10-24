#pragma once

// RC-QP primitives for the CollectiveBus data plane, lifted from the M0
// micro_ibv_smoke recipe that measured 107 Gb/s per lane and 196 Gb/s
// concurrent on this fabric. Library discipline: no std::exit — every
// failure returns a reason so startup errors stay legible.

#include <cstdint>
#include <string>

#include <infiniband/verbs.h>

#include "net/bus_types.hpp"

namespace dgpp::net {

// The RC transition working set, measured on this fabric (M0: 107 Gb/s per
// lane, 196 Gb/s concurrent) and shared by every QP the project creates —
// the bus's lanes and the transport regression probe alike, so a retune
// cannot drift between the deployment and the tool that validates it.
inline constexpr int kQpPsn = 0x1234;
inline constexpr int kQpTimeout = 14;
inline constexpr int kQpRetryCount = 7;
inline constexpr int kQpRnrRetry = 7;
inline constexpr int kQpMinRnrTimer = 12;
inline constexpr int kQpMaxRdAtomic = 1;

// One opened RoCE device: ACTIVE non-IB port, RoCEv2 GID, protection domain.
// Shared by every lane on the same physical function; the destructor closes
// when the last reference is dropped (the bus owns one per device name).
class VerbsDevice {
 public:
  // `name_hint` matches by exact name or substring; empty takes the first
  // device with an ACTIVE non-InfiniBand port.
  VerbsDevice(const std::string& name_hint, std::string* error);
  ~VerbsDevice();

  VerbsDevice(const VerbsDevice&) = delete;
  VerbsDevice& operator=(const VerbsDevice&) = delete;

  const std::string& name() const { return name_; }
  bool ok() const { return ctx_ != nullptr && pd_ != nullptr; }
  int port() const { return port_; }
  int gid_index() const { return gid_idx_; }
  const ibv_gid& gid() const { return gid_; }
  ibv_pd* pd() const { return pd_; }
  ibv_context* ctx() const { return ctx_; }
  // The active port's line rate in Gb/s (sysfs `ports/N/rate`; 0 when it
  // could not be read) — the bus derives its bulk pacing from it.
  double port_rate_gbps() const { return rate_gbps_; }

  // Caller owns the MR. Errors include the device, buffer size, CUDA view and
  // memlock state captured before registration; successful attempts log at DEBUG.
  ibv_mr* register_memory(void* buffer, size_t bytes, int access,
                         const char* purpose, std::string* error) const;

 private:
  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  std::string name_;
  int port_ = 1;
  double rate_gbps_ = 0.0;
  int gid_idx_ = 0;
  ibv_gid gid_{};
};

// One lane endpoint to one peer: a pinned slab registered as one MR, plus
// two RC QPs — one per slot pool — so every receive queue is
// type-predictable (latency traffic can never consume a bulk receive
// buffer or vice versa). Credit RDMA WRITEs ride the pool's QP unsignaled.
// The QP transitions and retry parameters are the measured working set
// from M0 (timeout 14, retries 7).
class RcLane {
 public:
  // Allocates the pinned slab (cudaHostAlloc) and registers it with remote
  // write so peer credit WRITEs can land in the completion cells.
  RcLane(VerbsDevice& device, const BusSlabLayout& layout, int qp_depth,
         std::string* error);
  ~RcLane();

  RcLane(const RcLane&) = delete;
  RcLane& operator=(const RcLane&) = delete;

  // RTR + RTS for the pool's QP against the peer's lane endpoint.
  bool connect(BusPool pool, const BusLaneEndpoint& peer, std::string* error);

  BusLaneEndpoint endpoint() const;

  uint8_t* slab() const { return slab_; }
  const BusSlabLayout& layout() const { return layout_; }
  bool ok() const {
    return slab_ != nullptr && qp_[0] != nullptr && qp_[1] != nullptr;
  }
  // The 64-byte doorbell and credit cells ride inline in their work
  // requests when the device grants max_inline_data >= 64 at QP creation
  // (2026-10-13; the NIC then reads the cell from the WQE instead of a
  // separate DMA of host memory — one PCIe round trip fewer per handshake).
  bool inline_cells(BusPool pool) const { return inline_ok_[pool == BusPool::kLatency ? 0 : 1]; }

  // -- send side ---------------------------------------------------------
  // Posts the payload SEND (unsignaled) followed by the doorbell SEND
  // (signaled) from the slot's send buffers. `seq` is the slot generation.
  // `ctl` is the COLLECTIVE sequence the doorbell belongs to (0 for
  // harness/flag traffic) — the per-collective kernels' claim gate
  // (see StartSlot::ctl; a blind claim eats a racing neighbor collective's
  // early doorbell — found by the M6 greedy-loop corruption hunt).
  // `payload_hash` is the sender's fold of the payload (0 = ungated,
  // harness/flag traffic) — the kernels' PLACEMENT gate (see
  // StartSlot::hash): a cell-polling kernel cannot lean on RC's CQE
  // ordering, so the doorbell itself must prove the payload landed.
  // `payload_local` overrides the payload's local address: any address in
  // `payload_mr` (e.g. a collective staging buffer) — only the doorbell's
  // remote ring position (the `slot` argument) participates in ring
  // discipline; the payload source is arbitrary MR-registered memory.
  bool post_send_pair(BusPool pool, uint32_t slot, uint32_t seq,
                      uint32_t len, std::string* error,
                      const void* payload_local = nullptr,
                      uint32_t payload_lkey = 0, uint32_t ctl = 0,
                      uint64_t payload_hash = 0);

  // RDMA WRITE of `len` bytes at `local` (any address in an MR of this
  // lane's device, `lkey`) into the peer's `remote_addr` (`rkey`): the
  // one-sided latency transport's row post (2026-10-19). Signaled when
  // asked — the caller spaces signals so the SQ retires (see
  // post_credit_write on the unsignaled SQ hazard).
  bool post_write(BusPool pool, const void* local, uint32_t lkey, uint32_t len,
                  uint64_t remote_addr, uint32_t rkey, bool signaled,
                  std::string* error);

  // Unsignaled 64-byte RDMA WRITE of the staging cell into the peer's
  // completion cell for the same pool/slot. No CQE at either side — the
  // sender observes it as a pinned flag. `seq` is the generation credited.
  bool post_credit_write(BusPool pool, uint32_t slot, uint32_t seq,
                         const BusLaneEndpoint& peer, uint64_t hash,
                         std::string* error);

  // -- receive side ------------------------------------------------------
  // Posts the payload and doorbell receive WRs for one slot (the credit
  // grant for that slot's next generation).
  bool post_recv_pair(BusPool pool, uint32_t slot, std::string* error);

  // Bounded drains. Returns the number of harvested entries; entries with
  // bad status are included and flagged — the caller decides failure.
  int poll_tx(BusPool pool, ibv_wc* out, int max);
  int poll_rx(BusPool pool, ibv_wc* out, int max);

  // TEMP hunt instrument (the burst-wedge hunt): one-line live QP state —
  // cur state, the last sent/received PSNs, and the RNR retry budget. A
  // wedged SQ shows up as sq_psn frozen below the posted count (an RNR
  // retry loop or an errored QP) while rq_psn keeps the arrivals' count.
  std::string qp_state_dump(BusPool pool) const;

 private:
  VerbsDevice* device_ = nullptr;
  ibv_qp* qp_[2] = {nullptr, nullptr};     // [latency, bulk]
  bool inline_ok_[2] = {false, false};
  ibv_cq* tx_cq_[2] = {nullptr, nullptr};
  ibv_cq* rx_cq_[2] = {nullptr, nullptr};
  ibv_mr* mr_ = nullptr;
  uint8_t* slab_ = nullptr;
  BusSlabLayout layout_{};
};

// Formats a verbs WC status for logs (status name + vendor error).
std::string bus_wc_error(const ibv_wc& wc);

}  // namespace dgpp::net
