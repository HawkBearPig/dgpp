#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/prefix_arena.hpp"
#include "sched/scheduler.hpp"

namespace dgpp {

// Request-owned prefill lifetime, shared by eager and graph execution. The
// cursor borrows its prompt, images and snapshot chain from this stable object.
// Adapters supply only the completion action (sampling and initial drafting).
template <class Model>
struct PrefillTask {
  using Progress = sched::SchedulerEngine::PrefillProgress;
  using Plan = sched::SchedulerEngine::PrefixPrefill;
  std::vector<int64_t> prompt, boundaries;
  std::vector<ImageInput> images;
  Plan plan;
  typename Model::SnapshotRequest snap, body_snap, head_snap;
  std::function<Progress(int64_t)> advance;
  std::function<Progress(int64_t, bool)> after;
  // Some diagnostic model facades do not implement resumable execution.
  std::shared_ptr<void> cursor;

  void commit(PrefixArena<Model>& arena) {
    if (snap.taken && !plan.snap_taken) {
      arena.commit(plan.snap_slot, snap);
      plan.snap_taken = true;
    }
    if (body_snap.taken && !plan.body_snap_taken) {
      arena.commit(plan.body_snap_slot, body_snap);
      plan.body_snap_taken = true;
    }
    if (head_snap.taken && !plan.head_snap_taken) {
      arena.commit(plan.head_snap_slot, head_snap);
      plan.head_snap_taken = true;
    }
  }

  template <class Finish>
  static std::unique_ptr<PrefillTask> begin(Model& model, PrefixArena<Model>& arena,
      int req, const std::vector<int64_t>& ids, int64_t reserve_tokens,
      int64_t chunk_tokens, const Plan& input, Finish finish) {
    auto task = std::make_unique<PrefillTask>();
    task->prompt = ids;
    if (input.boundaries) task->boundaries = *input.boundaries;
    if (input.images) task->images = *input.images;
    task->plan = input;
    task->plan.boundaries = &task->boundaries;
    task->plan.images = &task->images;
    typename Model::SnapshotRequest* snap = nullptr;
    if (input.snap_slot >= 0) {
      task->snap = arena.request(input.snap_slot, input.snap_position);
      snap = &task->snap;
    }
    if (input.body_snap_slot >= 0) {
      task->body_snap = arena.request(input.body_snap_slot, input.body_snap_position);
      task->body_snap.next = snap;
      snap = &task->body_snap;
    }
    if (input.head_snap_slot >= 0) {
      task->head_snap = arena.request(input.head_snap_slot, input.head_snap_position);
      task->head_snap.next = snap;
      snap = &task->head_snap;
    }
    // At the slot's position, or inside it on a family that opens on a
    // snapshot's first tokens (the arena refuses any other difference).
    if (input.attach_slot >= 0) arena.attach(req, input.attach_slot, input.attach_position);
    auto cursor = std::make_shared<typename Model::PrefillCursor>([&] {
      if constexpr (requires { model.session_prefill_begin(req, task->prompt, reserve_tokens,
          chunk_tokens, task->boundaries, snap, input.attach_position, &task->images); }) {
        return model.session_prefill_begin(req, task->prompt, reserve_tokens, chunk_tokens,
            task->boundaries, snap, input.attach_position, &task->images);
      } else {
        if (!task->images.empty()) throw std::logic_error("prefill: image execution cannot yield");
        return model.session_prefill_begin(req, task->prompt, reserve_tokens, chunk_tokens,
            task->boundaries, snap, input.attach_position);
      }
    }());
    if constexpr (requires { cursor->final_output_only; }) cursor->final_output_only = true;
    task->cursor = cursor;
    task->after = [cursor, task = task.get(), &arena, finish](int64_t start, bool done) {
      task->commit(arena);
      Progress progress;
      progress.computed_tokens = cursor->next - start;
      progress.snap_taken = task->plan.snap_taken;
      progress.body_snap_taken = task->plan.body_snap_taken;
      progress.head_snap_taken = task->plan.head_snap_taken;
      if (done) progress.first_token = finish(task->prompt, cursor->output);
      return progress;
    };
    task->advance = [cursor, task = task.get(), &model](int64_t budget) {
      const int64_t start = cursor->next;
      const bool done = model.session_prefill_advance(*cursor, budget);
      return task->after(start, done);
    };
    return task;
  }
};

}  // namespace dgpp
