# Review of MOONSHOT-1

Reviewed against revision `78038ed4` on 2026-10-30, using the proposed plan,
the implementation, recorded measurements, and NVIDIA CUDA documentation.
This was a static review; no hardware probes or benchmarks were run.

I would revise this before treating it as a buildable twelve-month plan.
The separate engine, retained fallback, correctness gates, and campaign
discipline are sound choices. The central claim—that the combined kernel
can approach the byte floor—still needs an experiment that tests the
proposed architecture as a whole.

These are the substantive findings, ordered by importance:

1. **[P1] The 38 MB lead is not yet an implementable capacity budget.**

   [MOONSHOT-1.md:110](MOONSHOT-1.md#L110) adds L2, shared memory, and
   registers as though they were one staging pool. Registers belong to
   threads; shared memory requires a defined CTA or cluster ownership
   scheme. A streamer on one SM cannot simply hand its register tile to a
   consumer on another. Phase-dependent roles could solve this, but the
   plan must specify where each tile lives and which threads eventually
   consume it. CUDA documents these
   [memory ownership boundaries](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#distributed-shared-memory).

   The budget also needs space for accumulators, activations, barriers, and
   overlapping tiles, with cache duplication accounted for. `__noinline__`
   alone does not establish that the combined kernel avoids spills.
   **Require a compiled kernel containing representative consumers and
   live staged weights, with measured registers, spills, occupancy, and
   usable lead, before accepting this capacity argument.**

2. **[P1] The stream model omits dependencies that determine which bytes to fetch.**

   [MOONSHOT-1.md:104](MOONSHOT-1.md#L104) says all bytes except routed
   experts are known before the step. Sparse attention is another
   exception: CSA2 computes selections from the current layer's query,
   then uses those selections to address KV. This is explicit in
   [csa2_layer.cu:726](src/models/dsv41/csa2_layer.cu#L726).

   Consequently, “160 µs of storage covers 70 µs of latency” does not prove
   sustained overlap. Useful future bytes must be available to fetch,
   space must become available in time, and the governor sometimes limits
   streaming to 150 GB/s. **Replace the fixed-gap synthetic gate with a
   dependency-aware replay of real layer traces, including router and
   selection release times, buffer occupancy, and governed bandwidth.**

3. **[P1] The 64-bit generation proposal conflicts with the unchanged wire protocol.**

   [MOONSHOT-1.md:435](MOONSHOT-1.md#L435) proposes 64-bit generations
   while earlier sections promise to preserve the wire format. The write
   trailer contains a 64-bit hash, a **32-bit generation**, and a 32-bit
   chunk index; the receive check explicitly reads that generation as
   `uint32_t`. See
   [bus_kernel.cu:1398](src/net/bus_kernel.cu#L1398). The host also
   truncates generations in
   [collective_bus.cpp:1642](src/net/collective_bus.cpp#L1642).

   **Choose a versioned wire change, a specified epoch/reset protocol, or
   the existing exhaustion-and-restart behavior.** Widening only the step
   descriptor does not solve wraparound. Add tests initialized near the
   boundary; a thousand-step soak cannot exercise it.

4. **[P1] Early launch needs a quiescence and cancellation protocol.**

   [MOONSHOT-1.md:121](MOONSHOT-1.md#L121) queues the next kernel before
   its descriptor exists. The plan also requires between-step fallback
   gathers, cache attachment, and prefill. It does not explain how an
   already queued kernel waiting for a descriptor gets out of the way
   when one of those operations must happen first.

   The existing engine explicitly drains before eager bus work, including
   fallback gathers:
   [graph_engine.hpp:2994](src/engine/graph_engine.hpp#L2994).
   **Define descriptor states for execute, skip/drain, and abort; buffer
   ownership; and the ordering of fallback and admission work.** Otherwise,
   a normal transition could become a dependency cycle that ends only at
   the watchdog deadline.

   PDL also provides opportunistic overlap, not guaranteed concurrency.
   Correctness must work with fully serialized execution, and the
   10–15 µs seam remains a measurement target.
   [NVIDIA's PDL contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

5. **[P1] Prefill coexistence needs an execution contract before template migration.**

   [MOONSHOT-1.md:341](MOONSHOT-1.md#L341) treats a block count as an SM
   partition. CUDA does not generally assign a requested subset of SMs
   merely because a grid has that many blocks; placement and available
   resources determine scheduling.
   [CUDA scheduling documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy)

   The scheduler already has a mixed-operation interface, but ordinary
   prefill advancement returns synchronously:
   [scheduler.cpp:883](src/sched/scheduler.cpp#L883). Specify whether the
   adapter uses that interface or introduces background prefill, and how
   completion, cancellation, workspaces, and collective ordering remain
   consistent.

   **Move mixed-workload acceptance ahead of each Phase 6 template
   switch.** Phase 7 currently comes after switching GLM-Flash away from
   its existing mixed verify behavior.

6. **[P1] Percent-of-floor is insufficient as the optimization and rollout criterion.**

   [MOONSHOT-1.md:24](MOONSHOT-1.md#L24) conflates required logical bytes
   with actual memory traffic. Caching, rereads, writes, and staging affect
   them differently. A slower implementation that reads more bytes could
   improve a poorly defined “percent-of-floor” metric.

   More importantly, cheaper speculative passes can produce fewer accepted
   tokens. The repository already records this: a split-K change reduced
   the code pass from 47.18 to 45.97 ms while decode throughput fell from
   76.1 to 75.0 tokens/s.
   [Recorded comparison](docs/model_cards/DeepSeek-V4.1-Flash-1017.md#split-k-on-the-narrow-attention-projections-2026-10-19)

   **Define logical work, measured traffic, and bandwidth efficiency
   separately; make decode tokens/s by class and concurrency a promotion
   gate.** Specify C8 targets explicitly—the referenced target table only
   defines C1. Also correct the “three to eight percent” statement: several
   listed targets are roughly 10–15% above their listed floors.

7. **[P2] The collective target assumes away measured rank jitter.**

   [MOONSHOT-1.md:268](MOONSHOT-1.md#L268) reduces skew to approximately
   3 µs because phase lengths become deterministic. The existing
   investigation reports about 15 µs median jitter on identical work,
   with substantial tails; it does not establish that removing launches
   eliminates it.
   [Skew measurements](docs/model_cards/DeepSeek-V4.1-Flash-1017.md#the-folds-spread-decomposed-2026-10-20)

   **Treat 12–15 µs as a hypothesis.** Test the fold after representative
   compute on all ranks, at actual batched H/5H payload sizes, and report
   latency distributions. An isolated small-payload collective cannot
   validate the full-step budget.

8. **[P2] The picker limitation is factually wrong.**

   [MOONSHOT-1.md:474](MOONSHOT-1.md#L474) interprets
   `DevicePicker::kSlots == 6` as a six-row limit. It counts buffered pick
   instances—the verify pick and draft picks—as explained in
   [tp_bus.hpp:1130](src/engine/tp_bus.hpp#L1130). The row limit is already
   `kPickMaxRows == 96` in
   [pick.hpp:35](src/kernels/pick.hpp#L35).

   Separate pick-instance capacity, request slots, and verification rows in
   the proposed ABI.

I would change the delivery order substantially. The parent proposal's
[one-layer experiment](MOONSHOT.md#L88) is the right first commitment:
real gate-up, down, hand-offs, and fold in one persistent kernel, compared
with the existing path at identical work. Add the actual staging/resource
scheme and test it on one and four GPUs. Establish explicit stop or
redesign criteria before building every format consumer and committing to
family ports.
