// RC-QP primitives for the CollectiveBus (see verbs.hpp).

#include "net/verbs.hpp"
#include "net/site_options.hpp"

#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "common/log.hpp"
#include "common/process_memory.hpp"

namespace dgpp::net {
namespace {

// The M0 working set, kept verbatim: these transition parameters measured
// 107 Gb/s single-lane and 196 Gb/s concurrent on this fabric. The values
// live in verbs.hpp (kQp*) so the transport regression probe shares them.

// First routable RoCEv2 GID index on the port, via sysfs. The v2 table
// entry sorts link-local (fe80::) first; switches forward link-local
// traffic unpredictably per port pair — the four-node mesh found directed
// pairs where fe80 GIDs were silently dropped (everything into one node
// died while its own outbound worked). The IPv4-mapped v2 GIDs (one per
// fabric subnet) route like normal unicast, so those are the ones to use.
int pick_rocev2_gid(ibv_context* ctx, int port) {
  std::string base = "/sys/class/infiniband/";
  base += ctx->device->name;
  base += "/ports/" + std::to_string(port) + "/gid_attrs/types/";
  int first_v2 = -1;
  for (int i = 0; i < 256; ++i) {
    FILE* f = std::fopen((base + std::to_string(i)).c_str(), "r");
    if (!f) continue;
    char type_buf[32] = {};
    const size_t r = std::fread(type_buf, 1, sizeof(type_buf) - 1, f);
    std::fclose(f);
    if (r == 0) continue;
    for (char* c = type_buf; *c; ++c)
      *c = static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
    if (std::strstr(type_buf, "v2") == nullptr) continue;
    if (first_v2 < 0) first_v2 = i;
    ibv_gid gid{};
    if (ibv_query_gid(ctx, port, i, &gid) != 0) continue;
    if (gid.raw[0] == 0xfe && gid.raw[1] == 0x80) continue;  // link-local
    return i;
  }
  // No routable v2 GID: fall back to the first v2 entry (matches the M0
  // tools' behavior, which measured link-local on these fabrics).
  return first_v2 < 0 ? 0 : first_v2;
}

}  // namespace

VerbsDevice::VerbsDevice(const std::string& name_hint, std::string* error) {
  int count = 0;
  ibv_device** list = ibv_get_device_list(&count);
  if (!list || count == 0) {
    if (list) ibv_free_device_list(list);
    *error = "no verbs devices present";
    return;
  }

  for (int i = 0; i < count && !ctx_; ++i) {
    const std::string candidate = list[i]->name;
    if (!name_hint.empty() && candidate != name_hint &&
        candidate.find(name_hint) == std::string::npos)
      continue;
    ibv_context* ctx = ibv_open_device(list[i]);
    if (!ctx) continue;

    ibv_device_attr attr{};
    if (ibv_query_device(ctx, &attr) != 0) {
      ibv_close_device(ctx);
      continue;
    }
    bool port_ok = false;
    for (int p = 1; p <= attr.phys_port_cnt; ++p) {
      ibv_port_attr pa{};
      if (ibv_query_port(ctx, p, &pa) == 0 && pa.state == IBV_PORT_ACTIVE &&
          pa.link_layer != IBV_LINK_LAYER_INFINIBAND) {
        port_ = p;
        port_ok = true;
        // The line rate, as sysfs renders it ("200 Gb/sec (4X HDR)").
        if (FILE* f = std::fopen(("/sys/class/infiniband/" + candidate +
                                  "/ports/" + std::to_string(p) + "/rate")
                                     .c_str(),
                                 "r")) {
          double gbps = 0.0;
          if (std::fscanf(f, "%lf", &gbps) == 1 && gbps > 0) rate_gbps_ = gbps;
          std::fclose(f);
        }
        break;
      }
    }
    if (!port_ok) {
      ibv_close_device(ctx);
      continue;
    }
    ctx_ = ctx;
    name_ = candidate;
  }
  ibv_free_device_list(list);

  if (!ctx_) {
    *error = "no ACTIVE RoCE device matching \"" + name_hint + "\"";
    return;
  }

  gid_idx_ = configured_gid_index(name_);
  if (gid_idx_ < 0) gid_idx_ = pick_rocev2_gid(ctx_, port_);
  if (ibv_query_gid(ctx_, port_, gid_idx_, &gid_) != 0) {
    *error = "query_gid failed on " + name_ + " idx=" + std::to_string(gid_idx_);
    ibv_close_device(ctx_);
    ctx_ = nullptr;
    return;
  }

  pd_ = ibv_alloc_pd(ctx_);
  if (!pd_) {
    *error = "alloc_pd failed errno=" + std::to_string(errno) + " on " + name_;
    ibv_close_device(ctx_);
    ctx_ = nullptr;
    return;
  }

  DGPP_LOG_INFO("verbs: device {} port {} gid_index {}", name_, port_, gid_idx_);
}

VerbsDevice::~VerbsDevice() {
  if (pd_) ibv_dealloc_pd(pd_);
  if (ctx_) ibv_close_device(ctx_);
}

ibv_mr* VerbsDevice::register_memory(void* buffer, size_t bytes, int access,
                                   const char* purpose, std::string* error) const {
  cudaPointerAttributes attrs{};
  const cudaError_t attr_error = cudaPointerGetAttributes(&attrs, buffer);
  const std::string visibility = attr_error == cudaSuccess
      ? "type=" + std::to_string(static_cast<int>(attrs.type)) +
            " device=" + std::to_string(attrs.device)
      : std::string("unqueryable (") + cudaGetErrorString(attr_error) + ")";
  const std::string before = memlock_status();
  DGPP_LOG_DEBUG("verbs: {} registration on {} bytes={} access={} cuda {}; before: {}",
                 purpose, name_, bytes, access, visibility, before);
  errno = 0;
  ibv_mr* mr = ibv_reg_mr(pd_, buffer, bytes, access);
  const int reg_errno = errno;
  if (!mr) {
    *error = std::string(purpose) + " registration failed on " + name_ +
             " errno=" + std::to_string(reg_errno) + " (" + std::strerror(reg_errno) +
             ") bytes=" + std::to_string(bytes) + " access=" + std::to_string(access) +
             " cuda " + visibility + "; before registration: " + before;
  } else {
    DGPP_LOG_DEBUG("verbs: {} registration on {} bytes={} -> ok", purpose, name_, bytes);
  }
  return mr;
}

RcLane::RcLane(VerbsDevice& device, const BusSlabLayout& layout, int qp_depth,
               std::string* error)
    : device_(&device), layout_(layout) {
  const cudaError_t alloc = cudaHostAlloc(reinterpret_cast<void**>(&slab_),
                                          layout.total_bytes,
                                          cudaHostAllocDefault);
  if (alloc != cudaSuccess || !slab_) {
    *error = "bus slab cudaHostAlloc failed (" +
             std::string(cudaGetErrorString(alloc)) +
             ") bytes=" + std::to_string(layout.total_bytes);
    slab_ = nullptr;
    return;
  }
  std::memset(slab_, 0, layout.total_bytes);

  // Remote write lets the peer's credit WRITEs land in the completion cells;
  // that is the only remote access the bus grants (no READ, no atomic).
  mr_ = device.register_memory(slab_, layout.total_bytes,
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
                               "bus slab", error);
  if (!mr_) return;

  // One CQ pair per pool's QP. Only doorbell SENDs are signaled and credit
  // WRITEs never are, so send-CQ load stays near one entry per message.
  for (int pool = 0; pool < 2; ++pool) {
    tx_cq_[pool] = ibv_create_cq(device.ctx(), qp_depth, nullptr, nullptr, 0);
    rx_cq_[pool] = ibv_create_cq(device.ctx(), qp_depth, nullptr, nullptr, 0);
    if (!tx_cq_[pool] || !rx_cq_[pool]) {
      *error = "create_cq failed errno=" + std::to_string(errno);
      return;
    }

    ibv_qp_init_attr init{};
    init.send_cq = tx_cq_[pool];
    init.recv_cq = rx_cq_[pool];
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = qp_depth;
    init.cap.max_recv_wr = qp_depth;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    // Ask for the 64-byte doorbell / credit cells inline (the driver writes
    // the granted size back into init.cap; below it the cells go by SGE as
    // before — a mismatch is a capability, not an error).
    init.cap.max_inline_data = static_cast<uint32_t>(sizeof(StartSlot));
    qp_[pool] = ibv_create_qp(device.pd(), &init);
    if (!qp_[pool]) {
      init.cap.max_inline_data = 0;
      qp_[pool] = ibv_create_qp(device.pd(), &init);
    }
    if (!qp_[pool]) {
      *error = "create_qp failed errno=" + std::to_string(errno);
      return;
    }
    inline_ok_[pool] = init.cap.max_inline_data >= sizeof(StartSlot);
    static const bool logged = [&] {
      DGPP_LOG_INFO("bus lane: inline doorbells {} (max_inline_data {} B, cell {} B)",
                    inline_ok_[pool] ? "on" : "off", init.cap.max_inline_data, sizeof(StartSlot));
      return true;
    }();
    (void)logged;

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = static_cast<uint8_t>(device.port());
    attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp_[pool], &attr,
                      IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX |
                          IBV_QP_ACCESS_FLAGS) != 0) {
      *error = "qp->INIT failed errno=" + std::to_string(errno);
      return;
    }
  }
}

RcLane::~RcLane() {
  for (int pool = 0; pool < 2; ++pool) {
    if (qp_[pool]) {
      DGPP_LOG_DEBUG("lane dtor: destroying qp pool={} qpn={}", pool,
                     qp_[pool]->qp_num);
      ibv_destroy_qp(qp_[pool]);
    }
    if (tx_cq_[pool]) {
      DGPP_LOG_DEBUG("lane dtor: destroying tx_cq pool={}", pool);
      ibv_destroy_cq(tx_cq_[pool]);
    }
    if (rx_cq_[pool]) {
      DGPP_LOG_DEBUG("lane dtor: destroying rx_cq pool={}", pool);
      ibv_destroy_cq(rx_cq_[pool]);
    }
  }
  if (mr_) {
    DGPP_LOG_DEBUG("lane dtor: dereg mr");
    ibv_dereg_mr(mr_);
  }
  if (slab_) {
    DGPP_LOG_DEBUG("lane dtor: freeing slab");
    cudaFreeHost(slab_);
    DGPP_LOG_DEBUG("lane dtor: slab freed");
  }
}

bool RcLane::connect(BusPool pool, const BusLaneEndpoint& peer,
                     std::string* error) {
  ibv_qp* qp = qp_[pool == BusPool::kLatency ? 0 : 1];
  const uint32_t peer_qpn =
      pool == BusPool::kLatency ? peer.qpn_lat : peer.qpn_bulk;
  const uint32_t peer_psn =
      pool == BusPool::kLatency ? peer.psn_lat : peer.psn_bulk;
  ibv_port_attr pa{};
  if (ibv_query_port(device_->ctx(), device_->port(), &pa) != 0) {
    *error = "query_port failed errno=" + std::to_string(errno);
    return false;
  }

  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = pa.active_mtu < IBV_MTU_4096 ? pa.active_mtu : IBV_MTU_4096;
  attr.dest_qp_num = peer_qpn;
  attr.rq_psn = peer_psn;
  attr.max_dest_rd_atomic = kQpMaxRdAtomic;
  attr.min_rnr_timer = kQpMinRnrTimer;
  attr.ah_attr.is_global = 1;
  std::memcpy(attr.ah_attr.grh.dgid.raw, peer.gid, 16);
  attr.ah_attr.grh.sgid_index = static_cast<uint8_t>(device_->gid_index());
  attr.ah_attr.grh.hop_limit = 255;
  attr.ah_attr.sl = 0;
  attr.ah_attr.port_num = static_cast<uint8_t>(device_->port());
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC |
                        IBV_QP_MIN_RNR_TIMER) != 0) {
    *error = "qp->RTR failed errno=" + std::to_string(errno);
    return false;
  }

  std::memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = kQpTimeout;
  attr.retry_cnt = kQpRetryCount;
  attr.rnr_retry = kQpRnrRetry;
  attr.sq_psn = kQpPsn;
  attr.max_rd_atomic = kQpMaxRdAtomic;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
    *error = "qp->RTS failed errno=" + std::to_string(errno);
    return false;
  }
  return true;
}

BusLaneEndpoint RcLane::endpoint() const {
  BusLaneEndpoint e{};
  e.qpn_lat = qp_[0]->qp_num;
  e.psn_lat = kQpPsn;
  e.qpn_bulk = qp_[1]->qp_num;
  e.psn_bulk = kQpPsn;
  std::memcpy(e.gid, device_->gid().raw, 16);
  e.rkey = mr_->rkey;
  e.slab_base = reinterpret_cast<uint64_t>(slab_);
  return e;
}

bool RcLane::post_send_pair(BusPool pool, uint32_t slot, uint32_t seq,
                            uint32_t len, std::string* error,
                            const void* payload_local, uint32_t payload_lkey,
                            uint32_t ctl, uint64_t payload_hash) {
  ibv_qp* qp = qp_[pool == BusPool::kLatency ? 0 : 1];
  uint8_t* payload = layout_.send_payload(slab_, pool, slot);
  StartSlot* doorbell = layout_.send_doorbell(slab_, pool, slot);
  // Publish order: hash, len and ctl first, seq last (readers acquire on
  // seq and then read them — the same discipline as FlagAck), payload
  // bytes precede all three. The hash rides the same doorbell DMA as
  // seq, so a claim that acquires seq sees the placement gate's target.
  doorbell->hash = payload_hash;
  doorbell->len = len;
  doorbell->ctl = ctl;
  std::atomic_thread_fence(std::memory_order_release);
  doorbell->seq = seq;
  std::atomic_thread_fence(std::memory_order_release);

  ibv_sge sge[2]{};
  sge[0].addr = reinterpret_cast<uint64_t>(
      payload_local != nullptr ? payload_local : payload);
  sge[0].lkey = payload_local != nullptr ? payload_lkey : mr_->lkey;
  sge[0].length = len;
  sge[1].addr = reinterpret_cast<uint64_t>(doorbell);
  sge[1].lkey = mr_->lkey;
  sge[1].length = sizeof(StartSlot);

  ibv_send_wr wr[2]{};
  wr[0].wr_id = bus_wr_id(pool, BusWr::kPayload, slot);
  wr[0].sg_list = &sge[0];
  wr[0].num_sge = 1;
  wr[0].opcode = IBV_WR_SEND;
  wr[0].next = &wr[1];
  wr[1].wr_id = bus_wr_id(pool, BusWr::kDoorbell, slot);
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;
  wr[1].opcode = IBV_WR_SEND;
  wr[1].send_flags = IBV_SEND_SIGNALED;
  if (inline_cells(pool)) wr[1].send_flags |= IBV_SEND_INLINE;  // the cell is copied at post time

  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(qp, &wr[0], &bad) != 0) {
    *error = "post_send pair failed errno=" + std::to_string(errno);
    return false;
  }
  return true;
}

bool RcLane::post_write(BusPool pool, const void* local, uint32_t lkey,
                        uint32_t len, uint64_t remote_addr, uint32_t rkey,
                        bool signaled, std::string* error) {
  ibv_qp* qp = qp_[pool == BusPool::kLatency ? 0 : 1];
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(local);
  sge.lkey = lkey;
  sge.length = len;
  ibv_send_wr wr{};
  wr.wr_id = bus_wr_id(pool, BusWr::kWrite, 0);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(qp, &wr, &bad) != 0) {
    *error = "row WRITE post failed errno=" + std::to_string(errno);
    return false;
  }
  return true;
}

bool RcLane::post_credit_write(BusPool pool, uint32_t slot, uint32_t seq,
                               const BusLaneEndpoint& peer, uint64_t hash,
                               std::string* error) {
  ibv_qp* qp = qp_[pool == BusPool::kLatency ? 0 : 1];
  BusCredit* staging = reinterpret_cast<BusCredit*>(
      layout_.credit_staging(slab_, pool, slot));
  // Publish order: hash and slot first, seq last (readers acquire on seq).
  staging->hash = hash;
  staging->slot = slot;
  std::atomic_thread_fence(std::memory_order_release);
  staging->seq = seq;
  std::atomic_thread_fence(std::memory_order_release);

  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(staging);
  sge.lkey = mr_->lkey;
  sge.length = sizeof(BusCredit);

  // The verbs contract retires an unsignaled WR's send-queue entry only at
  // the next signaled completion on the QP — a QP that posts only unsignaled
  // credit writes would fill its SQ monotonically and die at qp_depth (found
  // by the window-32 flood at exactly 1024 posts). Signal every kCreditSignal
  // -th generation: bounded SQ occupancy plus real error visibility for the
  // credit path, at a small fraction of the CQ load.
  constexpr uint32_t kCreditSignalEvery = 16;
  ibv_send_wr wr{};
  wr.wr_id = bus_wr_id(pool, BusWr::kCredit, slot);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = (seq % kCreditSignalEvery == 0) ? IBV_SEND_SIGNALED : 0;
  if (inline_cells(pool)) wr.send_flags |= IBV_SEND_INLINE;
  wr.wr.rdma.remote_addr =
      layout_.completion_remote_addr(peer.slab_base, pool, slot);
  wr.wr.rdma.rkey = peer.rkey;

  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(qp, &wr, &bad) != 0) {
    *error = "credit WRITE post failed errno=" + std::to_string(errno);
    return false;
  }
  return true;
}

bool RcLane::post_recv_pair(BusPool pool, uint32_t slot, std::string* error) {
  ibv_qp* qp = qp_[pool == BusPool::kLatency ? 0 : 1];
  ibv_sge payload_sge{};
  payload_sge.addr =
      reinterpret_cast<uint64_t>(layout_.recv_payload(slab_, pool, slot));
  payload_sge.lkey = mr_->lkey;
  payload_sge.length = static_cast<uint32_t>(layout_.pool_slot_bytes(pool));
  ibv_recv_wr payload_wr{};
  payload_wr.wr_id = bus_wr_id(pool, BusWr::kPayload, slot);
  payload_wr.sg_list = &payload_sge;
  payload_wr.num_sge = 1;
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(qp, &payload_wr, &bad) != 0) {
    *error = "post_recv payload failed errno=" + std::to_string(errno);
    return false;
  }

  ibv_sge doorbell_sge{};
  doorbell_sge.addr = reinterpret_cast<uint64_t>(
      layout_.recv_doorbell(slab_, pool, slot));
  doorbell_sge.lkey = mr_->lkey;
  doorbell_sge.length = sizeof(StartSlot);
  ibv_recv_wr doorbell_wr{};
  doorbell_wr.wr_id = bus_wr_id(pool, BusWr::kDoorbell, slot);
  doorbell_wr.sg_list = &doorbell_sge;
  doorbell_wr.num_sge = 1;
  bad = nullptr;
  if (ibv_post_recv(qp, &doorbell_wr, &bad) != 0) {
    *error = "post_recv doorbell failed errno=" + std::to_string(errno);
    return false;
  }
  return true;
}

std::string RcLane::qp_state_dump(BusPool pool) const {
  ibv_qp_attr attr{};
  ibv_qp* q = qp_[pool == BusPool::kLatency ? 0 : 1];
  if (!q) return "qp=null";
  ibv_qp_init_attr ia{};
  int mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RQ_PSN;
  if (ibv_query_qp(q, &attr, mask, &ia) != 0)
    return "query failed errno=" + std::to_string(errno);
  const char* state = "?";
  switch (attr.qp_state) {
    case IBV_QPS_RESET: state = "RESET"; break;
    case IBV_QPS_INIT: state = "INIT"; break;
    case IBV_QPS_RTR: state = "RTR"; break;
    case IBV_QPS_RTS: state = "RTS"; break;
    case IBV_QPS_SQD: state = "SQD"; break;
    case IBV_QPS_SQE: state = "SQE"; break;
    case IBV_QPS_ERR: state = "ERR"; break;
    default: break;
  }
  return std::string(state) + " sq_psn=" + std::to_string(attr.sq_psn) +
         " rq_psn=" + std::to_string(attr.rq_psn) +
         " ack_timeout=" + std::to_string(attr.timeout) +
         " retry_cnt=" + std::to_string(attr.retry_cnt) +
         " rnr_retry=" + std::to_string(attr.rnr_retry) +
         " min_rnr=" + std::to_string(attr.min_rnr_timer) +
         " sq_draining=" + std::to_string(attr.sq_draining);
}

int RcLane::poll_tx(BusPool pool, ibv_wc* out, int max) {
  return ibv_poll_cq(tx_cq_[pool == BusPool::kLatency ? 0 : 1], max, out);
}

int RcLane::poll_rx(BusPool pool, ibv_wc* out, int max) {
  return ibv_poll_cq(rx_cq_[pool == BusPool::kLatency ? 0 : 1], max, out);
}

std::string bus_wc_error(const ibv_wc& wc) {
  return "wc status=" + std::to_string(static_cast<int>(wc.status)) + " (" +
         ibv_wc_status_str(wc.status) + ") opcode=" +
         std::to_string(static_cast<int>(wc.opcode)) +
         " vendor_err=" + std::to_string(wc.vendor_err);
}

}  // namespace dgpp::net
