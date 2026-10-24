#pragma once
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "common/prefill_progress.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/pool_exhausted.hpp"
#include "engine/row_batch.hpp"

namespace dgpp {
// Shared batching policy and request commits. The session owns positions,
// snapshots and draft history; its executor owns only the physical row walk.
template <class Core>
struct PrefillBatch {
  template <class Cursor>
  static std::vector<typename Core::Outputs> run(Core& core,
      const std::vector<Cursor*>& cursors, const std::vector<int64_t>& budgets,
      const std::vector<int>& reqs, const std::vector<std::vector<int64_t>>& feeds, int graph_rows = 0) {
    using Outputs = typename Core::Outputs;
    const int snapshot_align_ = core.session_snapshot_align();
    const int max_requests_ = core.max_session_requests(), max_tokens_ = core.max_tokens();
    const int max_decode_rows_ = core.max_decode_rows();
    const int64_t max_context_ = core.max_context(), vocab_size_ = core.prefill_vocab_size();
    const int hidden_ = core.prefill_hidden_size(), lm_vocab_count_ = core.lm_vocab_count();

    using Executor = std::remove_reference_t<decltype(core.derived())>;
    if (reqs.empty() ? !Executor::kPrefillGroupAdvance : !Executor::kMixedPrefillVerify)
      throw std::logic_error("prefill batch: layer executor does not support this row layout");
    if (cursors.empty() || cursors.size() != budgets.size())
      throw std::invalid_argument("session_prefill_verify: one budget per prefill cursor");
    if (reqs.size() != feeds.size())
      throw std::invalid_argument("session_prefill_verify: one feed per decoding request");
    if (graph_rows < 0 || graph_rows > kSpecRows)
      throw std::invalid_argument("session_prefill_verify: invalid graph row count");
    std::vector<int> readers;
    std::vector<int64_t> counts;
    int64_t total_budget = 0;
    bool separate_readers = false;
    for (size_t s = 0; s < cursors.size(); ++s) {
      if (!cursors[s]) throw std::invalid_argument("session_prefill_verify: null cursor");
      const auto& cursor = *cursors[s];
      core.validate_prefill_cursor(cursor, "session_prefill_verify");
      if (!reqs.empty()) core.derived().validate_mixed_prefill_cursor(cursor);
      if constexpr (requires { cursor.images; })
        separate_readers |= cursor.images && !cursor.images->empty();
      if (std::find(readers.begin(), readers.end(), cursor.req) != readers.end())
        throw std::invalid_argument("session_prefill_verify: duplicate reader");
      readers.push_back(cursor.req);
      const int64_t budget = budgets[s] == 0 ? cursor.budget_tokens : budgets[s];
      if (budget < snapshot_align_ || budget > max_tokens_ || budget % snapshot_align_ != 0)
        throw std::invalid_argument("session_prefill_verify: invalid prefill budget");
      total_budget += budget;
      counts.push_back((reqs.empty() ? std::min(cursor.span_end(), cursor.next + budget)
                                    : cursor.next_end(budget)) - cursor.next);
    }
    std::vector<int64_t> ids, positions;
    std::vector<int32_t> spans, lengths;
    std::vector<int> offsets;
    std::vector<uint8_t> outputs(reqs.size(), 1), first(reqs.size(), 1), last(reqs.size(), 1);
    for (size_t s = 0; s < reqs.size(); ++s) {
      const int req = reqs[s];
      const bool padded = graph_rows > 0 && feeds[s].empty();
      if (!padded) core.check_req(req, "session_prefill_verify");
      if (!padded && (std::find(readers.begin(), readers.end(), req) != readers.end() ||
                      std::find(reqs.begin(), reqs.begin() + s, req) != reqs.begin() + s))
        throw std::invalid_argument("session_prefill_verify: duplicate request");
      const int n = padded ? graph_rows : static_cast<int>(feeds[s].size());
      const int64_t pos = padded ? -1 : core.session_pos_[req];
      if (n < 1 || n > kSpecRows || (!padded && (pos <= 0 || pos + n > max_context_)) ||
          (graph_rows > 0 && n != graph_rows))
        throw std::invalid_argument("session_prefill_verify: decode outside session bounds");
      for (const int64_t id : feeds[s])
        if (id < 0 || id >= vocab_size_)
          throw std::invalid_argument("session_prefill_verify: invalid token");
      offsets.push_back(static_cast<int>(ids.size()));
      if (padded) ids.insert(ids.end(), n, 0);
      else ids.insert(ids.end(), feeds[s].begin(), feeds[s].end());
      spans.push_back(padded ? -1 : req);
      positions.push_back(pos);
      lengths.push_back(n);
    }
    const int decode_rows = static_cast<int>(ids.size());
    const size_t live_decoders = std::count_if(feeds.begin(), feeds.end(), [](const auto& f) { return !f.empty(); });
    if (decode_rows > max_decode_rows_ || live_decoders + cursors.size() > static_cast<size_t>(max_requests_))
      throw std::invalid_argument("session_prefill_verify: too many decode rows or spans");
    // Budgets bound reader work; scratch bounds the entire physical walk.
    // Subtracting the decoder prefix from a smaller reader budget can waste
    // an aligned block even when that block still fits in scratch. Short
    // tails donate their unused shares before physical capacity trims a reader.
    int64_t prefill_rows = std::accumulate(counts.begin(), counts.end(), int64_t{0});
    const int64_t target = std::min<int64_t>(total_budget, max_tokens_ - decode_rows);
    while (prefill_rows > target) {
      auto largest = std::max_element(counts.begin(), counts.end());
      if (*largest <= snapshot_align_) break;
      const int64_t remainder = *largest % snapshot_align_;
      const int64_t take = remainder > 0 ? remainder : snapshot_align_;
      *largest -= take;
      prefill_rows -= take;
    }
    if (prefill_rows + decode_rows > max_tokens_)
      throw std::invalid_argument("session_prefill_verify: no room for prefill rows");
    for (size_t s = 0; s < cursors.size(); ++s) {
      const auto& cursor = *cursors[s];
      ids.insert(ids.end(), cursor.ids + cursor.next - cursor.start,
                           cursor.ids + cursor.next + counts[s] - cursor.start);
      spans.push_back(cursor.req);
      positions.push_back(cursor.next);
      lengths.push_back(static_cast<int32_t>(counts[s]));
      outputs.push_back(!cursor.final_output_only || cursor.next + counts[s] == cursor.end);
      first.push_back(cursor.span_start);
      last.push_back(cursor.closes_span(cursor.next + counts[s]));
    }
    for (size_t s = 0; s < reqs.size(); ++s)
      if (positions[s] >= 0 && core.derived().has_pool() && !core.derived().pool().ensure_request_blocks(reqs[s], positions[s] + lengths[s], core.stream_))
        throw std::runtime_error("session_prefill_verify: cache pool exhausted");
    for (auto* cursor : cursors) {
      if (!cursor->suspended) continue;
      core.push_position(cursor->req);
      if (core.mtp_) core.push_mtp_position(cursor->req);
      cursor->suspended = false;
    }
    RowRun run;
    run.req = cursors.front()->req;
    run.ids = ids.data();
    run.T = static_cast<int>(ids.size());
    run.span_reqs = spans.data();
    run.span_pos0 = positions.data();
    run.span_lens = lengths.data();
    run.num_spans = static_cast<int>(spans.size());
    run.output_spans = outputs.data();
    run.span_first_chunk = first.data();
    run.span_last_chunk = last.data();
    run.decode_spans = static_cast<int>(reqs.size());
    run.device_decode_outputs = graph_rows > 0;
    run.snapshots = true;
    // A lone prompt's chunk names itself and its successor (2026-10-18; the
    // families' chunk-ahead table staging under the scheduler, which walks
    // every prefill through here): its host ids by the prompt's own pointer
    // (the staging's key, stable across ticks) and the next chunk at this
    // budget on the cursor's cuts, as prefill_chunk() names it. The same
    // rows the batch copied above; a changed budget next tick just drops
    // the staging.
    if (reqs.empty() && cursors.size() == 1) {
      const auto& cursor = *cursors[0];
      const int64_t budget = budgets[0] == 0 ? cursor.budget_tokens : budgets[0];
      run.ids = cursor.ids + (cursor.next - cursor.start);
      run.pos0 = cursor.next;
      const int64_t c1 = cursor.next + counts[0];
      if (c1 < cursor.end) {
        int64_t c2 = cursor.end;
        for (const int64_t cut : cursor.cuts)
          if (cut > c1) c2 = std::min(c2, cut);
        if (budget > 0) c2 = cursor.relative_budget ? c1 + std::min(c2 - c1, budget)
                                                    : std::min(c2, (c1 / budget + 1) * budget);
        if (c2 > c1 && c2 - c1 <= max_tokens_) {
          run.next_ids = cursor.ids + (c1 - cursor.start);
          run.next_T = static_cast<int>(c2 - c1);
          run.next_pos0 = c1;
        }
      }
    }
    if (separate_readers || !core.derived().prefill_batch_fits(run)) {
      if (!reqs.empty()) throw std::invalid_argument("mixed walk exceeds continuation workspace");
      for (size_t s = 0; s < cursors.size(); ++s) core.derived().session_prefill_advance(*cursors[s], budgets[s]);
      return {};
    }
    core.derived().prepare_mixed_verify(0);
    for (auto* cursor : cursors)
      if constexpr (requires { core.derived().prepare_prefill_cursor(*cursor); })
        core.derived().prepare_prefill_cursor(*cursor);
    Outputs all = core.derived().run_rows(run);
    const auto slice = [&](int first, int count) {
      Outputs out;
      out.lm_vocab_begin = all.lm_vocab_begin;
      out.lm_vocab_count = all.lm_vocab_count;
      out.logits.assign(all.logits.begin() + static_cast<size_t>(first) * lm_vocab_count_,
                         all.logits.begin() + static_cast<size_t>(first + count) * lm_vocab_count_);
      out.final_hidden_bits.assign(all.final_hidden_bits.begin() + static_cast<size_t>(first) * hidden_,
                                  all.final_hidden_bits.begin() + static_cast<size_t>(first + count) * hidden_);
      return out;
    };
    std::vector<Outputs> out;
    for (size_t s = 0; s < reqs.size(); ++s) {
      if (graph_rows == 0) {
        out.push_back(slice(offsets[s], lengths[s]));
        core.session_pos_[reqs[s]] += lengths[s];
        core.push_position(reqs[s]);
      }
    }
    if (decode_rows) core.decode_rows_ = decode_rows;
    int hidden_offset = decode_rows;
    for (size_t s = 0; s < cursors.size(); ++s) {
      auto& cursor = *cursors[s];
      const int64_t c0 = cursor.next, c1 = c0 + counts[s];
      cursor.output = outputs[reqs.size() + s] ?
          slice((graph_rows > 0 ? 0 : decode_rows) + static_cast<int>(s), 1) : Outputs{};
      core.session_pos_[cursor.req] = c1;
      core.push_position(cursor.req);
      if (core.mtp_) {
        const int64_t r1 = std::min(c1, cursor.end - 1);
        if (r1 > c0) {
          if constexpr (requires { core.mtp_prefill_rows(cursor.req, c0, r1, cursor.ids, hidden_offset); })
            core.mtp_prefill_rows(cursor.req, c0, r1, cursor.ids + c0 + 1 - cursor.start, hidden_offset);
          else
            core.mtp_prefill_rows(cursor.req, c0, r1, cursor.ids + c0 + 1 - cursor.start);
        }
        core.mtp_pos_[cursor.req] = std::max(core.mtp_pos_[cursor.req], r1);
        core.push_mtp_position(cursor.req);
      }
      for (auto* snap = cursor.snap; snap; snap = snap->next) {
        if (snap->taken || snap->position != c1) continue;
        try {
          *snap->meta = core.session_snapshot(cursor.req, snap->dst);
          snap->taken = true;
        } catch (const CachePoolExhausted& e) {
          DGPP_LOG_WARN("prefix cache: snapshot at {} skipped for slot {}: {}", c1, cursor.req, e.what());
        }
      }
      if (cursor.cut_index < cursor.cuts.size() && cursor.cuts[cursor.cut_index] == c1) ++cursor.cut_index;
      cursor.span_start = last[reqs.size() + s];
      cursor.next = c1;
      if (c1 != cursor.end || graph_rows > 0) {
        DGPP_CUDA_OK(cudaMemsetAsync(core.d_session_pos_ + cursor.req, 0xff, sizeof(int64_t), core.stream_));
        if (core.mtp_) DGPP_CUDA_OK(cudaMemsetAsync(core.d_mtp_pos_ + cursor.req, 0xff, sizeof(int64_t), core.stream_));
        cursor.suspended = true;
      }
      hidden_offset += static_cast<int>(counts[s]);
      core.report_prefill_progress(cursor.req, c1);
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(core.stream_));
    return out;

  }
};
}  // namespace dgpp
