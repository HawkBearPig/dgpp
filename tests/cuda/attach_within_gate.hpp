#pragma once
// The attach inside a snapshot (the prefix cache's hit inside an entry), as
// one gate for every family that declares attaches_within_snapshot(): a slot
// opened on the first p tokens of a deeper snapshot is the slot opened on a
// snapshot the same walk took at p, bitwise. Compared: the suffix's last row
// and the row after it (a step, or the draft block's row under mtp). The
// positions: inside the first block, on a block boundary, inside a later
// whole block of the snapshot, and inside its partial block. The deeper
// snapshot is untouched by the requests opened inside it, and every block
// returns to the pool with the snapshots.
//
// `m`: two session slots or more, walks of at most 64 rows per chunk, a cache
// of at least eight blocks. `tokens(n, seed)`: n ids of the family's
// vocabulary.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace dgpp::test {

template <class Model, class Tokens>
void attach_within_gate(Model& m, const Tokens& tokens, int64_t vocab, bool mtp, const char* family) {
  const auto require = [family](bool cond, const std::string& what) {
    if (cond) return;
    std::fprintf(stderr, "FAILED (%s): %s\n", family, what.c_str());
    std::exit(1);
  };
  const auto bitwise = [](const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
  };
  const auto argmax = [](const std::vector<float>& row, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
      if (row[static_cast<size_t>(i)] > row[static_cast<size_t>(best)]) best = i;
    return static_cast<int64_t>(best);
  };
  require(m.attaches_within_snapshot(), "the family declares the attach inside a snapshot");
  const int V = m.lm_vocab_count();
  const int64_t B = m.kv_block_tokens();
  require(B >= 32, "the gate's positions assume blocks of at least 32 tokens");
  const int64_t P = 2 * B + 44;
  const std::vector<int64_t> L = tokens(static_cast<int>(P), 0xA24BAED4963EE407ull);
  const std::vector<int64_t> tail = tokens(20, 0x2545F4914F6CDD1Dull);
  const size_t bytes = m.session_snapshot_bytes();
  uint8_t* arena[2] = {nullptr, nullptr};
  for (uint8_t*& p : arena)
    require(cudaMalloc(reinterpret_cast<void**>(&p), bytes) == cudaSuccess, "arena");
  const auto row_after = [&](int req, const typename Model::Outputs& last) {
    const int64_t pending = argmax(last.logits, V);
    return mtp ? m.session_draft(req, std::vector<int64_t>{pending}) : m.session_step(req, pending);
  };
  for (const int64_t p : {B / 2 - 4, B, B + 12, 2 * B + 14}) {
    const std::string at = " at " + std::to_string(p);
    typename Model::SessionSnapshotMeta cut, deep;
    typename Model::SnapshotRequest snap;
    snap.position = p;
    snap.dst = arena[0];
    snap.meta = &cut;
    const typename Model::Outputs walk = m.session_prefill(0, L, std::vector<int64_t>{p}, &snap);
    require(snap.taken && cut.position == p, "the walk's own snapshot" + at);
    deep = m.session_snapshot(0, arena[1]);
    require(deep.position == P, "the deeper snapshot stands at the walk's end");
    const typename Model::Outputs walk_next = row_after(0, walk);
    m.session_close(0);
    // The request shares tokens [0, p] with the walk and parts from it after.
    std::vector<int64_t> suffix{L[static_cast<size_t>(p)]};
    suffix.insert(suffix.end(), tail.begin(), tail.end());
    if (suffix[1] == L[static_cast<size_t>(p) + 1]) suffix[1] = (suffix[1] + 1) % vocab;
    struct Run {
      typename Model::Outputs last, next;
    };
    const auto run = [&](bool within) {
      if (within) m.session_attach_within(1, deep, p);
      else m.session_attach(1, arena[0], cut);
      require(m.session_position(1) == p, "the slot opens at the position" + at);
      if (mtp) require(m.session_draft_position(1) == p, "the draft stands level with the session" + at);
      Run r;
      r.last = m.session_prefill_resume(1, suffix, {});
      r.next = row_after(1, r.last);
      m.session_close(1);
      return r;
    };
    const Run want = run(false), got = run(true);
    require(bitwise(got.last.logits, want.last.logits), "the suffix's last row differs from the cut's" + at);
    require(bitwise(got.next.logits, want.next.logits) && got.next.final_hidden_bits == want.next.final_hidden_bits,
            "the row after the suffix differs from the cut's" + at);
    // The deeper snapshot still opens the walk it was taken from.
    m.session_attach(1, arena[1], deep);
    const typename Model::Outputs again = row_after(1, walk);
    require(bitwise(again.logits, walk_next.logits) && again.final_hidden_bits == walk_next.final_hidden_bits,
            "the deeper snapshot changed under the requests" + at);
    m.session_close(1);
    // Outside the snapshot: refused, the slot left closed.
    for (const int64_t bad : {int64_t{0}, P, P + 1}) {
      bool refused = false;
      try {
        m.session_attach_within(1, deep, bad);
      } catch (const std::invalid_argument&) {
        refused = true;
      }
      require(refused && m.session_position(1) == 0, "a position outside the snapshot is refused");
    }
    m.session_release_snapshot(cut);
    m.session_release_snapshot(deep);
    require(m.kv_blocks_in_use() == 0, "every block released with the snapshots" + at);
  }
  for (uint8_t* p : arena) cudaFree(p);
  std::printf("[ OK ] %s: attach inside a snapshot == attach at a cut of the same walk, bitwise, at four positions%s\n",
              family, mtp ? " (draft block on)" : "");
}


// The same claim over rows a decode wrote, the shape of a cached answer: a
// walk prefills a prompt and decodes thirty tokens across a block boundary,
// keeping a live snapshot at six positions on the way and one at the end. A
// slot opened at each of the six inside the last snapshot is the slot
// opened on the walk's own snapshot there, bitwise. Then two slots open
// inside the last snapshot at once, the snapshot is released under them,
// and both still read what they read alone.
//
// `m`: three session slots, walks of at most 64 rows per chunk, a cache of
// at least 32 blocks. Under mtp the draft block trails the session by one
// row at every snapshot (the form that resumes with any token).
template <class Model, class Tokens>
void attach_within_decode_gate(Model& m, const Tokens& tokens, int64_t vocab, bool mtp, const char* family) {
  const auto require = [family](bool cond, const std::string& what) {
    if (cond) return;
    std::fprintf(stderr, "FAILED (%s, decode): %s\n", family, what.c_str());
    std::exit(1);
  };
  const auto bitwise = [](const typename Model::Outputs& a, const typename Model::Outputs& b) {
    return a.logits.size() == b.logits.size() &&
           (a.logits.empty() || std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0) &&
           a.final_hidden_bits == b.final_hidden_bits;
  };
  const int V = m.lm_vocab_count();
  const auto argmax = [V](const typename Model::Outputs& o) {
    int best = 0;
    for (int i = 1; i < V; ++i)
      if (o.logits[static_cast<size_t>(i)] > o.logits[static_cast<size_t>(best)]) best = i;
    return static_cast<int64_t>(best);
  };
  const auto row_after = [&](int req, const typename Model::Outputs& last) {
    const int64_t pending = argmax(last);
    return mtp ? m.session_draft(req, std::vector<int64_t>{pending}) : m.session_step(req, pending);
  };
  const int64_t B = m.kv_block_tokens();
  const int64_t P0 = B - 12, K = 30, P = P0 + K;
  const size_t bytes = m.session_snapshot_bytes();
  for (const uint64_t seed : {0x9E3779B97F4A7C15ull, 0xD1B54A32D192ED03ull}) {
    std::vector<int64_t> ids = tokens(static_cast<int>(P0), seed);
    const std::vector<int64_t> tail = tokens(20, seed ^ 0x2545F4914F6CDD1Dull);
    const std::vector<int64_t> spots{P0, P0 + 5, B - 1, B, B + 1, P - 1};
    std::vector<uint8_t*> arena(spots.size() + 1, nullptr);
    for (uint8_t*& p : arena)
      require(cudaMalloc(reinterpret_cast<void**>(&p), bytes) == cudaSuccess, "arena");
    std::vector<typename Model::SessionSnapshotMeta> metas(spots.size() + 1);
    typename Model::Outputs out = m.session_prefill(0, ids);
    for (int64_t q = P0; q < P; ++q) {
      const int64_t pending = argmax(out);
      ids.push_back(pending);
      for (size_t i = 0; i < spots.size(); ++i)
        if (spots[i] == q) metas[i] = m.session_snapshot(0, arena[i]);
      if (mtp) (void)m.session_draft(0, std::vector<int64_t>{pending});
      out = m.session_step(0, pending);
    }
    ids.push_back(argmax(out));  // the token the walk would write next
    typename Model::SessionSnapshotMeta& deep = metas.back();
    deep = m.session_snapshot(0, arena.back());
    require(deep.position == P, "the last snapshot stands at the walk's end");
    m.session_close(0);
    struct Run {
      typename Model::Outputs last, next;
    };
    const auto suffix_at = [&](int64_t q) {
      std::vector<int64_t> suffix{ids[static_cast<size_t>(q)]};
      suffix.insert(suffix.end(), tail.begin(), tail.end());
      if (suffix[1] == ids[static_cast<size_t>(q) + 1]) suffix[1] = (suffix[1] + 1) % vocab;
      return suffix;
    };
    std::vector<Run> alone(spots.size());
    for (size_t i = 0; i < spots.size(); ++i) {
      const int64_t q = spots[i];
      const std::string at = " at " + std::to_string(q);
      const auto run = [&](bool within) {
        if (within) m.session_attach_within(1, deep, q);
        else m.session_attach(1, arena[i], metas[i]);
        require(m.session_position(1) == q, "the slot opens at the position" + at);
        Run r;
        r.last = m.session_prefill_resume(1, suffix_at(q), {});
        r.next = row_after(1, r.last);
        m.session_close(1);
        return r;
      };
      const Run want = run(false);
      alone[i] = run(true);
      require(bitwise(alone[i].last, want.last), "the suffix's last row differs from the walk's own snapshot's" + at);
      require(bitwise(alone[i].next, want.next), "the row after the suffix differs from the walk's own snapshot's" + at);
    }
    // Two slots inside the last snapshot at once, on either side of the
    // block boundary; the snapshot released while both are open.
    const size_t a = 1, b = 4;
    m.session_attach_within(1, deep, spots[a]);
    m.session_attach_within(2, deep, spots[b]);
    const typename Model::Outputs last_a = m.session_prefill_resume(1, suffix_at(spots[a]), {});
    const typename Model::Outputs last_b = m.session_prefill_resume(2, suffix_at(spots[b]), {});
    m.session_release_snapshot(deep);
    const typename Model::Outputs next_b = row_after(2, last_b);
    const typename Model::Outputs next_a = row_after(1, last_a);
    require(bitwise(last_a, alone[a].last) && bitwise(last_b, alone[b].last),
            "two slots inside one snapshot: a suffix's last row differs from the slot's alone");
    require(bitwise(next_a, alone[a].next) && bitwise(next_b, alone[b].next),
            "two slots inside one snapshot, released under them: the next row differs from the slot's alone");
    m.session_close(1);
    m.session_close(2);
    for (size_t i = 0; i < spots.size(); ++i) m.session_release_snapshot(metas[i]);
    require(m.kv_blocks_in_use() == 0, "every block released with the snapshots and the slots");
    for (uint8_t* p : arena) cudaFree(p);
  }
  std::printf("[ OK ] %s: attach inside a snapshot over decoded rows == attach at the walk's own snapshot, bitwise, "
              "at six positions; two slots inside one released snapshot%s\n",
              family, mtp ? " (draft block on)" : "");
}

}  // namespace dgpp::test
