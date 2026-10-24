#!/usr/bin/env python3
"""Run serve_load with reconciled scheduler counters around every phase.

Requires an otherwise idle server. Accept an idle snapshot only when its
prompt/token deltas equal the completed requests' usage, with a five-second
deadline. Polling is outside the client and engine timing intervals.
Arguments and JSON fields are the same as serve_load, plus each phase's
engine counters and decode tokens/s. Warmup and isolation are reconciled too.

Two rates come out of every phase, and the tables keep them apart:
  decode tok/s  (engine)  tokens generated / the engine's decode-step time —
                prefill and queueing excluded; the headline throughput, the
                figure docs/benchmarks.md reports by concurrency.
  wall tok/s    (client)  tokens generated / the request group's wall time —
                prefill and queueing included; the service-level view.
The per-class table printed first is the decode rate; serve_load's own
table (the wall rate) follows it.
"""
import math
import time

import serve_load
from serve_prefill_probe import get


COUNTERS = ("step_ms", "prefill_ms", "decode_steps", "decode_rows", "tokens_generated",
            "prompts_prefilled", "prompt_tokens", "prompt_tokens_computed")


def reconciled_metrics(host, port, before=None, requests=(), *, timeout=5):
    """Read immediately; retry only while the known work has not been published."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("metrics timeout must be finite and positive")
    expected = {"prompts_prefilled": len(requests),
                "tokens_generated": sum(r["tokens"] for r in requests)}
    deadline = time.monotonic() + timeout
    observed = None
    metrics = None

    def failure(reason):
        detail = (f"expected delta {expected}, observed delta {observed}"
                  if before is not None else "expected an idle scheduler")
        gauges = ({k: metrics[k] for k in ("active", "queued")}
                  if metrics is not None else None)
        return RuntimeError(f"scheduler counters {reason}; {detail}; active/queued={gauges}")

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise failure("did not reconcile before the deadline")
        try:
            metrics = get(host, port, "/v1/metrics", timeout=remaining)["scheduler"]
        except TimeoutError as error:
            raise failure("timed out reading metrics") from error
        if before is not None:
            observed = {k: metrics[k] - before[k] for k in expected}
            if any(value < 0 for value in observed.values()):
                raise failure("regressed (server restart or reset)")
            if any(observed[k] > expected[k] for k in expected):
                raise failure("include unexpected requests or tokens")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise failure("did not reconcile before the deadline")
        if (not metrics["active"] and not metrics["queued"] and
                (before is None or observed == expected)):
            return metrics
        # This only limits polling frequency. Readiness requires exact counters,
        # even if multiple consecutive reads return the same stale snapshot.
        time.sleep(min(0.01, remaining))


class TimedLoad:
    def __init__(self):
        self.previous = None
        self.decode_rates = []  # (class, concurrency, engine tok/s) per measured phase

    def settle(self, host, port, requests):
        self.previous = reconciled_metrics(host, port, self.previous, requests)
        return self.previous

    def phase(self, host, port, model, c, max_tokens, think, prompt_base,
              prompts=None, temperature=0, label="", reports=None):
        before = self.settle(host, port, [])
        measured = []
        rate = serve_load.phase(host, port, model, c, max_tokens, think, prompt_base,
                                prompts, temperature, label, measured)
        result = measured[-1]
        after = self.settle(host, port, result["requests"])
        delta = {name: after[name] - before[name] for name in COUNTERS}
        tokens = delta["tokens_generated"] - delta["prompts_prefilled"]
        if tokens <= 0 or not math.isfinite(delta["step_ms"]) or delta["step_ms"] <= 0:
            raise RuntimeError("no measurable decode work")
        result["engine"] = {**delta, "decode_tokens": tokens,
                            "tokens_per_s": tokens * 1000 / delta["step_ms"]}
        if reports is not None:
            reports.append(result)
        self.decode_rates.append((label.strip() or "mixed", c, result["engine"]["tokens_per_s"]))
        print(f"   engine decode: {result['engine']['tokens_per_s']:.3f} tok/s (decode steps only; the wall rate above includes prefill)", flush=True)
        return rate


def decode_table(rates):
    """Median decode-only rate per (class, concurrency) over the repetitions, as a markdown table."""
    import statistics
    cells = {}
    for cls, c, rate in rates:
        cells.setdefault((cls, c), []).append(rate)
    classes = list(dict.fromkeys(cls for cls, _, _ in rates))
    cs = sorted({c for _, c, _ in rates})
    lines = ["== per-class engine decode tokens/s (decode steps only; prefill and queueing excluded)",
             "| class | " + " | ".join(f"c={c}" for c in cs) + " |", "|---|" + "---|" * len(cs)]
    for cls in classes:
        lines.append(f"| {cls} | " + " | ".join(f"{statistics.median(cells[(cls, c)]):.1f}" if (cls, c) in cells else "—" for c in cs) + " |")
    return "\n".join(lines)


def main():
    load = TimedLoad()
    serve_load.main(phase_runner=load.phase, settle=load.settle)
    if load.decode_rates:
        print(decode_table(load.decode_rates), flush=True)


if __name__ == "__main__":
    main()
