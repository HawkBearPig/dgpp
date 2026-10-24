#pragma once

// CollectiveBus shared layout and wire types (M5 deliverable 2, DESIGN §6).
//
// One QP per (peer, lane). Traffic on that QP in each direction is strictly
// {payload SEND, doorbell SEND} pairs, so every receive queue is
// type-predictable: a doorbell receive completion means the whole message
// landed (RC ordering delivers the payload CQE first).
//
// The credit grant travels the opposite direction as an unsignaled 64-byte
// RDMA WRITE into the sender's registered completion cells — the same
// pinned-flag visibility contract as the GPU acks, without consuming a
// receive work request or generating a CQE.

#include <cstddef>
#include <cstdint>

#include "kernels/flag_protocol_types.hpp"

// The write ring's geometry helpers are shared by the engine (host) and
// the kernel (device).
#if defined(__CUDACC__)
#define DGPP_BUS_HD __host__ __device__
#else
#define DGPP_BUS_HD
#endif

namespace dgpp::net {

// ---- work-request id codec ---------------------------------------------------
// CQEs carry no QP identity; the wr_id packs everything the poller needs.
//   bits  0..31  slot index within the pool
//   bits 32..39  pool kind (kPoolLatency / kPoolBulk)
//   bits 40..43  wr kind (kWrPayload / kWrDoorbell / kWrCredit)
//   bits 44..63  reserved
enum class BusPool : uint64_t { kLatency = 0, kBulk = 1 };
enum class BusWr : uint64_t { kPayload = 0, kDoorbell = 1, kCredit = 2, kWrite = 3 };

inline uint64_t bus_wr_id(BusPool pool, BusWr kind, uint32_t slot) {
  return (static_cast<uint64_t>(slot) & 0xffffffffULL) |
         (static_cast<uint64_t>(pool) << 32) |
         (static_cast<uint64_t>(kind) << 40);
}

inline uint32_t bus_wr_slot(uint64_t wr_id) {
  return static_cast<uint32_t>(wr_id & 0xffffffffULL);
}

inline BusPool bus_wr_pool(uint64_t wr_id) {
  return static_cast<BusPool>((wr_id >> 32) & 0xffULL);
}

inline BusWr bus_wr_kind(uint64_t wr_id) {
  return static_cast<BusWr>((wr_id >> 40) & 0xfULL);
}

// ---- message integrity -------------------------------------------------------
// The bus carries a 64-bit consumer result through the credit WRITE; it does
// not interpret it. The transport harness consumer folds payloads with this
// mixing function and the sender mirrors it — the same golden-ratio pair mix
// the flag protocol uses. Payload length must be a multiple of 8.
constexpr uint64_t kFoldMultiplier = 0x9E3779B97F4A7C15ULL;

// The positional fold: (word + position + 1) * golden-ratio, XOR-combined.
// The plain XOR-of-(w*M) canceled identical words — a uniform 4-word
// payload folded to 0 exactly like all-zeros, and the 2026-09-01 hunt
// watched a stale uniform payload pass a placement gate meant to catch it.
// The position mix breaks every uniform-cancel; the XOR combine keeps the
// result order-independent and bit-replicable on host and device (the
// kernels' strided partials XOR the same per-position contributions).
// Payload length must be a multiple of 8.
inline uint64_t bus_fold64(const uint64_t* words, size_t word_count) {
  uint64_t hash = 0;
  for (size_t i = 0; i < word_count; ++i)
    hash ^= (words[i] + i + 1) * kFoldMultiplier;
  return hash;
}

// ---- rendezvous exchange frame (little-endian, version 2) --------------------
// Full-mesh bring-up: every rank's frame lists its QP endpoints for every
// OTHER rank (peer entries ordered by peer rank ascending), so the listener
// (rank 0) can distribute everyone's frame to everyone — TP=4 needs no
// second rendezvous mechanism. Slot geometry must match on all sides; a
// mismatch is a config error reported over TCP with a legible reason
// before any QP transitions (roster precedent: bare closes hide bugs).
//
// header (32 bytes):
//   u32 magic | u8 version | u8 lane_count | u8 peer_count | u8 reserved |
//   i32 rank | u32 lat_slots | u32 lat_slot_bytes | u32 bulk_slots |
//   u32 bulk_slot_bytes | u32 reserved2
// lane entry (48 bytes):
//   u32 qpn_lat | u32 psn_lat | u32 qpn_bulk | u32 psn_bulk |
//   u8 gid[16] | u32 rkey | u32 reserved | u64 slab_base

constexpr uint32_t kBusExchangeMagic = 0x53424744;  // 'DGBS' little-endian
constexpr uint32_t kBusTableMagic = 0x54424744;     // 'DGBT' little-endian
constexpr uint32_t kBusErrorMagic = 0x52424744;     // 'DGBR' little-endian
// Post-connect barrier frames (4 bytes each). Every rank reports ready once
// its QPs are RTS with the initial receives posted; rank 0 answers GO once
// it has every ready. Without this a fast rank's first RDMA WRITE can land
// on a peer QP that is still in INIT — the IB CM's RTR-before-RTU rule,
// re-learned the hard way (the 2026-09-02 seq-1 loopback wedge).
constexpr uint32_t kBusReadyMagic = 0x59424744;     // 'DGBY' little-endian
constexpr uint32_t kBusGoMagic = 0x4f424744;        // 'DGBO' little-endian
// Version 2 includes a final u32 in collective placement hashes. Reject
// older peers before they can disagree on the visibility gate. Version 3
// (2026-10-19) appends the write ring to the slab layout.
constexpr uint8_t kBusExchangeVersion = 3;
constexpr size_t kBusExchangeHeaderBytes = 32;
constexpr size_t kBusExchangeLaneBytes = 48;
constexpr size_t kBusMaxLanes = 2;
constexpr size_t kBusMaxPeers = 3;  // TP=4 world
constexpr size_t kBusMaxExchangeBytes =
    kBusExchangeHeaderBytes +
    kBusMaxPeers * kBusMaxLanes * kBusExchangeLaneBytes;

// Per-lane identity of one endpoint: everything the peer needs to
// transition its QPs to RTR and to address our completion cells with RDMA
// WRITEs. Each pool has its own QP so every receive queue stays
// type-predictable — one QP per (peer, lane, pool) keeps latency and bulk
// traffic from consuming each other's differently-sized receive buffers.
struct BusLaneEndpoint {
  uint32_t qpn_lat = 0;
  uint32_t psn_lat = 0x1234;
  uint32_t qpn_bulk = 0;
  uint32_t psn_bulk = 0x1234;
  uint8_t gid[16] = {};
  uint32_t rkey = 0;
  uint64_t slab_base = 0;  // NIC-visible address of our pinned slab
};

// One rank's full endpoint set. `peers[peer_index][lane]` where peer_index
// is the position of the peer rank among all ranks except ours, ascending.
struct BusRankExchange {
  int32_t rank = -1;
  uint8_t lane_count = 0;
  uint8_t peer_count = 0;
  uint32_t lat_slots = 0;
  uint32_t lat_slot_bytes = 0;
  uint32_t bulk_slots = 0;
  uint32_t bulk_slot_bytes = 0;
  BusLaneEndpoint peers[kBusMaxPeers][kBusMaxLanes] = {};
};

// Position of `my_rank` among the peers listed in `peer_rank`'s exchange
// frame (all ranks except `peer_rank`, ascending). The peer's entry for us
// sits at this index in its table.
inline size_t bus_peer_index(int my_rank, int peer_rank) {
  return static_cast<size_t>(my_rank > peer_rank ? my_rank - 1 : my_rank);
}

size_t bus_rank_exchange_encode(const BusRankExchange& in, uint8_t* out,
                                size_t cap);
bool bus_rank_exchange_decode(const uint8_t* in, size_t len,
                              BusRankExchange* out);

// ---- slab layout --------------------------------------------------------------
// One pinned allocation per (peer, lane), registered as one MR with remote
// write enabled. Both roles (send structures and receive structures) live in
// it; the peer's credit WRITEs target our completion cells, whose address it
// computes from our slab_base plus these offsets — identical layout constants
// on both sides, versioned by the exchange frame.
//
//   [ send slots: lat (bytes+doorbell) | bulk (bytes+doorbell) ]
//   [ completion cells: lat | bulk ]          <- peer credit WRITEs land here
//   [ recv payload slots: lat | bulk ]        <- receive WRs target these
//   [ recv doorbell cells: lat | bulk ]
//   [ ack cells: lat | bulk ]                 <- consumer kernel publishes
//   [ credit staging: lat | bulk ]            <- our credit WRITE sources
//   [ control cell ]                          <- orderly kernel stop
//   [ write ring: kBusWriteRing rows ]        <- the peer's collective rows
//                                                land here by RDMA WRITE
//
// The write ring (2026-10-19, the one-sided latency transport): the
// decode step's collectives no longer travel as SEND pairs claimed through
// door cells, acked and credited. The sender's kernel lays its row out in
// 512-byte CHUNKS — 31 lanes of 16 bytes of bf16 (248 elements) and one
// trailer lane {hash, generation, chunk} — the engine RDMA-WRITEs the row
// into every peer's ring at row (generation - 1) % kBusWriteRing, and the
// receiving kernel folds the chunks as they prove: a chunk is consumed
// the moment its 16-byte trailer's generation matches and its hash folds
// from the 31 data lanes as read. No ordering is assumed between the NIC's
// writes of different 16-byte units (the fabric showed a 64-byte line's
// last unit visible before its first, 2026-10-19); a torn chunk fails its
// hash and is re-read. The warps poll at a gentle cadence (the poll is
// the data read): on this fabric a NIC write's one-way latency grows with
// the GPU's poll rate on its destination, whatever each poll reads. Ring reuse needs no credit: a rank starts
// generation g + kBusWriteRing only after its generation g + 1 completed,
// which needed every peer's g + 1 row, which every peer wrote only after
// finishing generation g — so every peer has consumed row g by then
// (kBusWriteRing >= 2 suffices; the depth is slack).

struct BusSlabLayout {
  size_t lat_slots = 0;
  size_t lat_slot_bytes = 0;
  size_t bulk_slots = 0;
  size_t bulk_slot_bytes = 0;

  size_t total_bytes = 0;
  // Offsets from slab base (all multiples of 64).
  size_t send_slots_off = 0;    // per slot: payload then 64B doorbell
  size_t completion_off = 0;    // one 64B cell per slot, latency pool first
  size_t recv_payload_off = 0;  // slot-strided by pool
  size_t recv_doorbell_off = 0;
  size_t ack_off = 0;
  size_t credit_staging_off = 0;
  size_t control_off = 0;
  size_t write_ring_off = 0;   // kBusWriteRing rows of write_row_bytes
  size_t write_row_bytes = 0;  // bus_write_row_bytes(lat_slot_bytes)

  size_t pool_slots(BusPool pool) const {
    return pool == BusPool::kLatency ? lat_slots : bulk_slots;
  }
  size_t pool_slot_bytes(BusPool pool) const {
    return pool == BusPool::kLatency ? lat_slot_bytes : bulk_slot_bytes;
  }
  // Address helpers. `base` is the slab start; every structure is 64B aligned.
  uint8_t* send_payload(uint8_t* base, BusPool pool, uint32_t slot) const;
  StartSlot* send_doorbell(uint8_t* base, BusPool pool, uint32_t slot) const;
  uint8_t* completion_cell(uint8_t* base, BusPool pool, uint32_t slot) const;
  uint8_t* recv_payload(uint8_t* base, BusPool pool, uint32_t slot) const;
  StartSlot* recv_doorbell(uint8_t* base, BusPool pool, uint32_t slot) const;
  FlagAck* ack_cell(uint8_t* base, BusPool pool, uint32_t slot) const;
  uint8_t* credit_staging(uint8_t* base, BusPool pool, uint32_t slot) const;
  StartSlot* control_cell(uint8_t* base) const;
  uint8_t* write_row(uint8_t* base, uint32_t row) const;

  // Remote completion-cell address for pool/slot, given the peer's slab base
  // (both sides use the same layout, so the peer computes ours identically).
  uint64_t completion_remote_addr(uint64_t peer_slab_base, BusPool pool,
                                  uint32_t slot) const;
  // The peer's write-ring row our collective WRITE for `row` targets.
  uint64_t write_row_remote_addr(uint64_t peer_slab_base, uint32_t row) const;
};

// The write ring's geometry (see the slab layout above).
constexpr uint32_t kBusWriteRing = 8;           // rows per (peer, lane) slab
constexpr uint32_t kBusWriteChunkBytes = 512;   // one warp: 32 lanes x 16 bytes
constexpr uint32_t kBusWriteChunkElems = 248;   // 31 data lanes x 8 bf16
constexpr uint32_t kBusWriteDataLanes = 31;     // lane 31 carries the trailer
inline uint32_t bus_write_chunks(size_t bf16_elems) {
  return static_cast<uint32_t>((bf16_elems + kBusWriteChunkElems - 1) / kBusWriteChunkElems);
}
// The chunked row for a payload of `payload_bytes` (a latency slot): every
// row of the ring and every staging row is this long.
inline size_t bus_write_row_bytes(size_t payload_bytes) {
  return static_cast<size_t>(bus_write_chunks(payload_bytes / 2)) * kBusWriteChunkBytes;
}
// Lane striping (2026-10-19): with two lanes (the Spark's two PCIe
// functions of one port) a row's chunks split into contiguous runs, run
// l written through lane l's queue pair into the peer's lane-l slab ring
// at the chunks' own offsets — the DMA of a row then crosses both x4
// links. [begin, end) of lane `lane`'s run; empty runs are legal.
DGPP_BUS_HD inline void bus_write_lane_run(uint32_t chunks, int lanes, int lane,
                                                   uint32_t* begin, uint32_t* end) {
  const uint32_t per = (chunks + static_cast<uint32_t>(lanes) - 1) / static_cast<uint32_t>(lanes);
  const uint32_t b = per * static_cast<uint32_t>(lane);
  *begin = b < chunks ? b : chunks;
  const uint32_t e = b + per;
  *end = e < chunks ? e : chunks;
}
// The write kernel's grid for a row (the engine waits for every block's
// ready bit): eight warps a block, one chunk round per warp when the row
// is short, at most 32 blocks.
constexpr uint32_t kBusWriteWarpsPerBlock = 8;
constexpr uint32_t kBusWriteMaxBlocks = 16;
DGPP_BUS_HD inline uint32_t bus_write_blocks(uint32_t chunks) {
  uint32_t blocks = (chunks + kBusWriteWarpsPerBlock - 1) / kBusWriteWarpsPerBlock;
  if (blocks == 0) blocks = 1;
  return blocks > kBusWriteMaxBlocks ? kBusWriteMaxBlocks : blocks;
}
DGPP_BUS_HD inline int bus_write_lane_of(uint32_t chunks, int lanes, uint32_t chunk) {
  const uint32_t per = (chunks + static_cast<uint32_t>(lanes) - 1) / static_cast<uint32_t>(lanes);
  return static_cast<int>(chunk / per);
}

// Derived geometry from raw counts; returns false when the counts are not
// usable (zero slots, slot bytes not a multiple of 64 or below the doorbell
// size). One layout object is shared by every lane so slot math is identical
// on both endpoints.
bool bus_slab_layout(size_t lat_slots, size_t lat_slot_bytes, size_t bulk_slots,
                     size_t bulk_slot_bytes, BusSlabLayout* out);

// The credit record that travels in the RDMA WRITE. Fields are published in
// order (hash, slot) with `seq` last, so a reader that acquire-loads `seq`
// sees a coherent cell — the same publish-last discipline as FlagAck.
// `seq` must equal the generation the sender posted into that slot;
// anything else is a protocol error surfaced loudly. `hash` is the
// consumer's result, opaque to the bus.
struct alignas(64) BusCredit {
  uint64_t hash = 0;
  uint32_t slot = 0;
  uint32_t seq = 0;
  uint32_t pad[10];
};
static_assert(sizeof(BusCredit) == 64, "BusCredit must occupy one cache line");

}  // namespace dgpp::net
