#!/usr/bin/env python3
"""DGPP side of the paired probe: exact per-request spec stats via metrics deltas.

Run after spec_ab_probe_prompts.py against the dgpp-serve that carries the
drafter under test; per-request alpha/tau come from the /metrics spec_decode
deltas around each single request. The prompt cells are imported from
spec_ab_probe_prompts, never restated: the pairing means nothing unless both
engines saw the same bytes.

Point --url at the dgpp-serve port (its /metrics and /v1/models are its own).
The listener must serve exactly one model and remain otherwise idle: a swap
proxy's model catalogue does not identify the owner of its current counters.
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_ab_probe_prompts import CELLS  # noqa: E402


def served_ids(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
        return sorted({m["id"] for m in json.load(r).get("data", [])})


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=120) as r:
        return json.load(r)["scheduler"]


def resolve_model(base, model):
    ids = served_ids(base)
    if len(ids) != 1:
        raise ValueError(f"{base} reports model ids {ids}: connect to the single-model engine's own port")
    if model is not None and model != ids[0]:
        raise ValueError(f"{base} does not serve '{model}' (it reports {ids})")
    return ids[0]


def idle_metrics(base, timeout=0, completed_after=None):
    deadline = time.monotonic() + timeout
    while True:
        result = metrics(base)
        if (not result["active"] and not result["queued"] and
                (completed_after is None or result["prompts_prefilled"] > completed_after)):
            return result
        if time.monotonic() >= deadline:
            raise RuntimeError("spec probe requires an otherwise idle engine with published completion counters")
        time.sleep(0.01)


def run(cells, out, base, model):
    results = []
    for c in cells:
        before = idle_metrics(base)
        m0 = before["spec_decode"]
        d0, a0 = m0["num_draft_tokens_total"], m0["num_accepted_tokens_total"]
        n0 = m0["num_drafts_total"]
        p0 = list(m0["num_draft_tokens_per_pos_total"])
        pa0 = list(m0["num_accepted_tokens_per_pos_total"])
        t0 = time.monotonic()
        body = {"model": model, "temperature": 0, "stream": False,
                "max_tokens": c["max_tokens"], "messages": c["messages"]}
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        wall = time.monotonic() - t0
        after = idle_metrics(base, timeout=5, completed_after=before["prompts_prefilled"])
        if after["prompts_prefilled"] - before["prompts_prefilled"] != 1:
            raise RuntimeError("spec counters do not describe exactly one request; check concurrent traffic")
        m1 = after["spec_decode"]
        d1, a1 = m1["num_draft_tokens_total"], m1["num_accepted_tokens_total"]
        p1 = list(m1["num_draft_tokens_per_pos_total"])
        pa1 = list(m1["num_accepted_tokens_per_pos_total"])
        drafts, accepted = d1 - d0, a1 - a0
        per_pos = [b - a for a, b in zip(pa0, pa1)]
        pos_drafts = [b - a for a, b in zip(p0, p1)]
        # The counter counts accepted drafts only. Each verification round
        # also emits one bonus/correction token, included in mean length tau.
        steps = m1["num_drafts_total"] - n0
        if min(drafts, accepted, steps, *per_pos, *pos_drafts) < 0:
            raise RuntimeError("spec counters reset during the request")
        tau = None if not steps else 1 + accepted / steps
        results.append({
            "cell": c["cell"], "model": model, "ok": True, "wall_s": round(wall, 2),
            "steps": steps, "drafts": drafts, "accepted": accepted,
            "alpha": None if drafts == 0 else round(accepted / drafts, 4),
            "tau": None if tau is None else round(tau, 3),
            "per_pos_accepted": per_pos, "per_pos_drafts": pos_drafts,
            "usage": resp.get("usage"),
        })
        print(f"{c['cell']:>16}  α={results[-1]['alpha']}  τ={results[-1]['tau']}  "
              f"steps={steps} drafts={drafts} accepted={accepted} wall={wall:.1f}s")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DGPP side of the paired spec probe.")
    ap.add_argument("out", help="where to write the per-cell JSON")
    ap.add_argument("--url", default="http://localhost:8000",
                    help="the dgpp-serve base URL (its own port when one is running)")
    ap.add_argument("--model", default=None, help="defaults to the single served id")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    try:
        model = resolve_model(base, a.model)
    except ValueError as error:
        sys.exit(str(error))
    print(f"# {len(CELLS)} cells against {base} as '{model}'")
    run(CELLS, a.out, base, model)
