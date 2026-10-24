// CollectiveBus shared layout and wire codecs (see bus_types.hpp).

#include "net/bus_types.hpp"

#include <cstring>

namespace dgpp::net {

namespace {

void put_u32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

void put_u64(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

uint32_t get_u32(const uint8_t* p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i)
    v |= static_cast<uint32_t>(p[i]) << (8 * i);
  return v;
}

uint64_t get_u64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

size_t round_up_64(size_t v) { return (v + 63) & ~size_t{63}; }

}  // namespace

size_t bus_rank_exchange_encode(const BusRankExchange& in, uint8_t* out,
                                size_t cap) {
  const size_t lanes = in.lane_count;
  const size_t peers = in.peer_count;
  const size_t total =
      kBusExchangeHeaderBytes + peers * lanes * kBusExchangeLaneBytes;
  if (lanes == 0 || lanes > kBusMaxLanes || peers > kBusMaxPeers ||
      total > cap)
    return 0;
  std::memset(out, 0, total);

  put_u32(&out[0], kBusExchangeMagic);
  out[4] = kBusExchangeVersion;
  out[5] = static_cast<uint8_t>(lanes);
  out[6] = static_cast<uint8_t>(peers);
  out[7] = 0;
  const uint32_t rank = static_cast<uint32_t>(in.rank);
  put_u32(&out[8], rank);
  put_u32(&out[12], in.lat_slots);
  put_u32(&out[16], in.lat_slot_bytes);
  put_u32(&out[20], in.bulk_slots);
  put_u32(&out[24], in.bulk_slot_bytes);
  // out[28..31] reserved

  for (size_t peer = 0; peer < peers; ++peer) {
    for (size_t lane = 0; lane < lanes; ++lane) {
      uint8_t* e = out + kBusExchangeHeaderBytes +
                   (peer * lanes + lane) * kBusExchangeLaneBytes;
      const BusLaneEndpoint& src = in.peers[peer][lane];
      put_u32(&e[0], src.qpn_lat);
      put_u32(&e[4], src.psn_lat);
      put_u32(&e[8], src.qpn_bulk);
      put_u32(&e[12], src.psn_bulk);
      std::memcpy(&e[16], src.gid, 16);
      put_u32(&e[32], src.rkey);
      put_u32(&e[36], 0);
      put_u64(&e[40], src.slab_base);
    }
  }
  return total;
}

bool bus_rank_exchange_decode(const uint8_t* in, size_t len,
                              BusRankExchange* out) {
  if (len < kBusExchangeHeaderBytes) return false;
  if (get_u32(&in[0]) != kBusExchangeMagic) return false;
  if (in[4] != kBusExchangeVersion) return false;
  const size_t lanes = in[5];
  const size_t peers = in[6];
  if (lanes == 0 || lanes > kBusMaxLanes) return false;
  if (peers == 0 || peers > kBusMaxPeers) return false;
  if (len != kBusExchangeHeaderBytes + peers * lanes * kBusExchangeLaneBytes)
    return false;

  *out = BusRankExchange{};
  out->rank = static_cast<int32_t>(get_u32(&in[8]));
  out->lane_count = static_cast<uint8_t>(lanes);
  out->peer_count = static_cast<uint8_t>(peers);
  out->lat_slots = get_u32(&in[12]);
  out->lat_slot_bytes = get_u32(&in[16]);
  out->bulk_slots = get_u32(&in[20]);
  out->bulk_slot_bytes = get_u32(&in[24]);

  for (size_t peer = 0; peer < peers; ++peer) {
    for (size_t lane = 0; lane < lanes; ++lane) {
      const uint8_t* e =
          in + kBusExchangeHeaderBytes +
          (peer * lanes + lane) * kBusExchangeLaneBytes;
      BusLaneEndpoint& dst = out->peers[peer][lane];
      dst.qpn_lat = get_u32(&e[0]);
      dst.psn_lat = get_u32(&e[4]);
      dst.qpn_bulk = get_u32(&e[8]);
      dst.psn_bulk = get_u32(&e[12]);
      std::memcpy(dst.gid, &e[16], 16);
      dst.rkey = get_u32(&e[32]);
      dst.slab_base = get_u64(&e[40]);
    }
  }
  return true;
}

bool bus_slab_layout(size_t lat_slots, size_t lat_slot_bytes, size_t bulk_slots,
                     size_t bulk_slot_bytes, BusSlabLayout* out) {
  if (lat_slots == 0 || bulk_slots == 0) return false;
  if (lat_slot_bytes < 64 || lat_slot_bytes % 64 != 0) return false;
  if (bulk_slot_bytes < 64 || bulk_slot_bytes % 64 != 0) return false;

  BusSlabLayout l;
  l.lat_slots = lat_slots;
  l.lat_slot_bytes = lat_slot_bytes;
  l.bulk_slots = bulk_slots;
  l.bulk_slot_bytes = bulk_slot_bytes;

  const size_t per_send_slot_lat = lat_slot_bytes + 64;
  const size_t per_send_slot_bulk = bulk_slot_bytes + 64;
  size_t off = 0;
  l.send_slots_off = off;
  off += lat_slots * per_send_slot_lat + bulk_slots * per_send_slot_bulk;
  off = round_up_64(off);
  l.completion_off = off;
  off += (lat_slots + bulk_slots) * 64;
  l.recv_payload_off = off;
  off += lat_slots * lat_slot_bytes + bulk_slots * bulk_slot_bytes;
  off = round_up_64(off);
  l.recv_doorbell_off = off;
  off += (lat_slots + bulk_slots) * 64;
  l.ack_off = off;
  off += (lat_slots + bulk_slots) * 64;
  l.credit_staging_off = off;
  off += (lat_slots + bulk_slots) * 64;
  l.control_off = off;
  off += 64;
  // The write ring: appended, so every offset above is the version-2
  // layout's (the exchange frame versions the whole).
  l.write_row_bytes = bus_write_row_bytes(lat_slot_bytes);
  l.write_ring_off = off;
  off += static_cast<size_t>(kBusWriteRing) * l.write_row_bytes;
  l.total_bytes = off;

  *out = l;
  return true;
}

namespace {

// Slot addressing shared by the send and receive structures. Send slots are
// laid out per-slot as [payload][doorbell]; receive pools are regions.
size_t send_slot_span(const BusSlabLayout& l, BusPool pool) {
  return l.pool_slot_bytes(pool) + 64;
}

}  // namespace

uint8_t* BusSlabLayout::send_payload(uint8_t* base, BusPool pool,
                                      uint32_t slot) const {
  uint8_t* region = base + send_slots_off;
  if (pool == BusPool::kBulk)
    region += lat_slots * (lat_slot_bytes + 64);
  return region + static_cast<size_t>(slot) * send_slot_span(*this, pool);
}

StartSlot* BusSlabLayout::send_doorbell(uint8_t* base, BusPool pool,
                                         uint32_t slot) const {
  return reinterpret_cast<StartSlot*>(send_payload(base, pool, slot) +
                                      pool_slot_bytes(pool));
}

uint8_t* BusSlabLayout::completion_cell(uint8_t* base, BusPool pool,
                                         uint32_t slot) const {
  uint8_t* region = base + completion_off;
  if (pool == BusPool::kBulk) region += lat_slots * 64;
  return region + static_cast<size_t>(slot) * 64;
}

uint8_t* BusSlabLayout::recv_payload(uint8_t* base, BusPool pool,
                                      uint32_t slot) const {
  uint8_t* region = base + recv_payload_off;
  if (pool == BusPool::kBulk) region += lat_slots * lat_slot_bytes;
  return region + static_cast<size_t>(slot) * pool_slot_bytes(pool);
}

StartSlot* BusSlabLayout::recv_doorbell(uint8_t* base, BusPool pool,
                                        uint32_t slot) const {
  uint8_t* region = base + recv_doorbell_off;
  if (pool == BusPool::kBulk) region += lat_slots * 64;
  return reinterpret_cast<StartSlot*>(region +
                                      static_cast<size_t>(slot) * 64);
}

FlagAck* BusSlabLayout::ack_cell(uint8_t* base, BusPool pool,
                                 uint32_t slot) const {
  uint8_t* region = base + ack_off;
  if (pool == BusPool::kBulk) region += lat_slots * 64;
  return reinterpret_cast<FlagAck*>(region + static_cast<size_t>(slot) * 64);
}

uint8_t* BusSlabLayout::credit_staging(uint8_t* base, BusPool pool,
                                        uint32_t slot) const {
  uint8_t* region = base + credit_staging_off;
  if (pool == BusPool::kBulk) region += lat_slots * 64;
  return region + static_cast<size_t>(slot) * 64;
}

StartSlot* BusSlabLayout::control_cell(uint8_t* base) const {
  return reinterpret_cast<StartSlot*>(base + control_off);
}

uint8_t* BusSlabLayout::write_row(uint8_t* base, uint32_t row) const {
  return base + write_ring_off + static_cast<size_t>(row) * write_row_bytes;
}

uint64_t BusSlabLayout::write_row_remote_addr(uint64_t peer_slab_base,
                                              uint32_t row) const {
  return peer_slab_base + write_ring_off +
         static_cast<uint64_t>(row) * write_row_bytes;
}

uint64_t BusSlabLayout::completion_remote_addr(uint64_t peer_slab_base,
                                                BusPool pool,
                                                uint32_t slot) const {
  uint64_t region = peer_slab_base + completion_off;
  if (pool == BusPool::kBulk) region += lat_slots * 64;
  return region + static_cast<uint64_t>(slot) * 64;
}

}  // namespace dgpp::net
