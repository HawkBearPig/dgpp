#!/usr/bin/env python3
"""DGPP side of the paired probe: exact per-request spec stats via metrics deltas.

Run after spec_ab_probe_prompts.py against the vLLM engine on the same
prompts; per-request alpha/tau come from the /metrics spec_decode deltas
around each single request.
"""
import json, time, urllib.request

def metrics():
    with urllib.request.urlopen("http://localhost:8000/metrics", timeout=120) as r:
        return json.load(r)["scheduler"]["spec_decode"]

def run(cells, out):
    results = []
    for c in cells:
        m0 = metrics()
        d0, a0 = m0["num_draft_tokens_total"], m0["num_accepted_tokens_total"]
        n0 = m0["num_drafts_total"]
        p0 = list(m0["num_draft_tokens_per_pos_total"])
        pa0 = list(m0["num_accepted_tokens_per_pos_total"])
        t0 = time.time()
        body = {"model": "Qwen3.8-Flash-Next", "temperature": 0, "stream": False,
                "max_tokens": c["max_tokens"], "messages": c["messages"]}
        req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        wall = time.time() - t0
        m1 = metrics()
        d1, a1 = m1["num_draft_tokens_total"], m1["num_accepted_tokens_total"]
        p1 = list(m1["num_draft_tokens_per_pos_total"])
        pa1 = list(m1["num_accepted_tokens_per_pos_total"])
        drafts, accepted = d1 - d0, a1 - a0
        per_pos = [b - a for a, b in zip(pa0, pa1)]
        pos_drafts = [b - a for a, b in zip(p0, p1)]
        # τ = accepted/(steps) : steps = num_new_drafts; accepted counts bonus? match bench: τ counts bonus token
        steps = m1["num_drafts_total"] - n0
        tau = None if not steps else accepted / steps
        results.append({
            "cell": c["cell"], "ok": True, "wall_s": round(wall, 2),
            "steps": steps, "drafts": drafts, "accepted": accepted,
            "alpha": None if drafts == 0 else round(accepted / drafts, 4),
            "tau": None if not steps else round(accepted / steps, 3),
            "per_pos_accepted": per_pos, "per_pos_drafts": pos_drafts,
            "usage": resp.get("usage"),
        })
        print(f"{c['cell']:>16}  α={results[-1]['alpha']}  τ={results[-1]['tau']}  "
              f"steps={steps} drafts={drafts} accepted={accepted} wall={wall:.1f}s")
    json.dump(results, open(out, "w"), indent=1)

if __name__ == "__main__":
    import sys
    code = open("/tmp/opencode/run_probes.py").read()
    ns = {}
    exec(code.split("results = []")[0].replace("URL = sys.argv[1]", "").replace("OUT = sys.argv[2]", ""), ns)
    run(ns["CELLS"], sys.argv[1])
