#!/usr/bin/env python3
"""The NVMe cold tier under a random workload with restarts (issue #26): a
handful of document families of random size, random questions, random
prompt_cache_key pins and prefix_cache opt-outs, every (document, question)
pair repeated at random so its greedy transcript is checked against the first
time it was answered — whether that answer came from a cold prefill, a memory
hit, a restore from disk or an adopted slab after a restart. Every
`--restart-every` requests the world is taken down and brought back under the
same configuration (dgpp-cluster), the launcher's op-stream identity line is
checked, and the adopted entries are counted.

Boot the world first (a deployment JSON with the tier on and a small
`engine.prefix_cache_gib`), then:

  scripts/serve_nvme_cache_fuzz.py --config that.json --bin build-ci/dgpp-serve \\
      --port 18080 --requests 60 --restart-every 20 --seed 1 --out report.json

Exit 0 when every repeat matched, no tier op failed and every restart's op
streams agreed; the report names each mismatch.

`--clients K` runs K requests at a time (the restarts and crashes happen
between rounds, with nothing in flight); `--cancel-pct P` closes P percent
of the requests' connections a random moment after they were sent (during
the restore or the prefill, mostly); `--crash-every N` kills one rank with
SIGKILL (`--crash-rank R`, or a random one) every N requests instead of a
clean restart, then brings the world down and up — the slab's adoption must
then agree on every rank or start fresh everywhere, and every answer
restored afterwards must still equal its first.

With more than one client the transcript check is advisory (reported, not
a finding) unless `--strict-transcripts`: readers in flight together share
the prefill budget, so a prompt's chunks shrink to 512, 256 or 128 tokens
and its cold walk's arithmetic differs from the solo walk's (the DSA
sites are not split-invariant) — the same answer variance as the first
prompt after a boot on the controller's ramp, and not the tier's doing.
The verdict then rests on the tier's counters, the adoption agreement at
every restart, the op-stream identity and the service having settled.

The deployment template pins the prefill chunk size (`engine.first_token_target_ms`
and `engine.decode_gap_target_ms` at 0, so every tick offers the idle budget):
a cold walk's arithmetic depends on its chunk shapes (the first prompt after a
boot walks the controller's 256-token ramp), and an answer restored from the
tier reproduces the walk that made the entry — the check must compare walks
of one shape. The driver warns when the served world leaves the targets on.
"""
import argparse
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_nvme_cache_check as chk  # noqa: E402
from serve_client import served_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def restart(config, bin_path, log_lines, crash_rank=None):
    if crash_rank is not None:
        # A crash: one rank dies mid-flight; the others exit on the lost peer
        # (the launcher's down reaps whatever is left).
        kill = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "dgpp-cluster"), "signal", "--config", config,
                               "--rank", str(crash_rank), "--signal", "KILL"], capture_output=True, text=True, timeout=120)
        log_lines.append(f"crash: SIGKILL to rank {crash_rank}, rc {kill.returncode}")
        time.sleep(5)
    down = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "dgpp-cluster"), "down", "--config", config],
                          capture_output=True, text=True, timeout=600)
    identical = "identical across" in down.stdout or "identical across" in down.stderr
    single = "on 1 node" in down.stdout or "op streams" not in down.stdout + down.stderr
    if crash_rank is not None:
        identical = True  # a killed rank's op stream is cut short; the digest handshake judges the slab
    log_lines.append(f"down: rc {down.returncode}, op streams {'identical' if identical else ('n/a' if single else 'DIFFER')}")
    up = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "dgpp-cluster"), "up", "--config", config]
                        + (["--bin", bin_path] if bin_path else []), capture_output=True, text=True, timeout=1500)
    ready = "READY" in up.stdout
    log_lines.append(f"up: rc {up.returncode}, {'ready' if ready else 'NOT READY'}")
    return (identical or single) and ready


def cancel(host, port, model, prompt, max_tokens, extra, after_ms):
    """Sends the request and drops the connection after `after_ms` without
    reading the answer: the service sees a client gone during the restore or
    the prefill (mostly) and must cancel cleanly."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "chat_template_kwargs": {"clear_thinking": False}}
    body.update(extra or {})
    conn = http.client.HTTPConnection(host, port, timeout=3600)
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        time.sleep(after_ms / 1000.0)
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--config", help="deployment JSON, for the restarts")
    ap.add_argument("--bin", default=None, help="the binary dgpp-cluster up stages (default: the release build)")
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--restart-every", type=int, default=0, help="0: no restarts")
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--clients", type=int, default=1, help="requests in flight at a time")
    ap.add_argument("--cancel-pct", type=float, default=0.0, help="percent of requests cancelled mid-flight")
    ap.add_argument("--crash-every", type=int, default=0, help="kill a rank every N requests (instead of a clean restart)")
    ap.add_argument("--crash-rank", default="random", help="the rank to kill, or random")
    ap.add_argument("--strict-transcripts", action="store_true",
                    help="count transcript mismatches as findings even with several clients")
    args = ap.parse_args()
    strict = args.clients <= 1 or args.strict_transcripts
    advisory = []
    rng = random.Random(args.seed)
    model = served_model(args.host, args.port)
    adm = chk.metrics(args.host, args.port)["service"]["admission"]
    if "nvme_cache" not in chk.metrics(args.host, args.port):
        print("NOTE: the world has no NVMe cache; running as a plain workload (service checks only)", flush=True)
    if adm.get("first_token_target_ms", 1) or adm.get("decode_gap_target_ms", 1):
        print("WARNING: the world's latency targets are on; cold repeats may differ by chunk shape "
              "(set engine.first_token_target_ms and engine.decode_gap_target_ms to 0 in the template)", flush=True)
    docs = ["".join(chk.filler(args.seed * 100 + i, rng.randint(250, 700))) for i in range(args.docs)]
    questions = [f"\nQuestion {i}: write a numbered list describing the first {30 + i} parcels and their cities."
                 for i in range(4)]
    expected = {}  # (doc, question) -> the first transcript
    findings, log_lines, steps = [], [], []
    restarts = 0
    lock = threading.Lock()
    crashes = 0

    def one(n, d, q, body_extra, cancel_after_ms):
        prompt = docs[d] + questions[q]
        t0 = time.perf_counter()
        if cancel_after_ms is not None:
            cancel(args.host, args.port, model, prompt, args.max_tokens, body_extra, cancel_after_ms)
            step = {"n": n, "doc": d, "question": q, "cancelled_after_ms": cancel_after_ms, "extra": body_extra,
                    "wall_ms": (time.perf_counter() - t0) * 1000}
            with lock:
                steps.append(step)
            print(f"{n:3d} doc {d} q {q} cancelled after {cancel_after_ms} ms {body_extra}", flush=True)
            return
        r = chk.ask(args.host, args.port, model, prompt, args.max_tokens, extra=body_extra)
        cached = r["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
        key = (d, q)
        step = {"n": n, "doc": d, "question": q, "ttft_ms": r["ttft_ms"], "cached": cached,
                "prompt_tokens": r["usage"].get("prompt_tokens"), "extra": body_extra,
                "wall_ms": (time.perf_counter() - t0) * 1000}
        with lock:
            if key in expected:
                step["repeat"] = True
                if r["text"] != expected[key]:
                    step["mismatch"] = True
                    (findings if strict else advisory).append(
                        {"n": n, "doc": d, "question": q, "what": "transcript differs from the first answer",
                         "first": expected[key][:200], "now": r["text"][:200]})
            else:
                expected[key] = r["text"]
            steps.append(step)
        print(f"{n:3d} doc {d} q {q} ttft {r['ttft_ms']:7.0f} ms cached {str(cached):>6} "
              f"{'repeat' if step.get('repeat') else 'first '} {'MISMATCH' if step.get('mismatch') else ''} {body_extra}",
              flush=True)

    pool = ThreadPoolExecutor(max_workers=max(1, args.clients))
    pending = []
    for n in range(args.requests):
        boundary = n > 0 and ((args.restart_every and n % args.restart_every == 0) or
                              (args.crash_every and n % args.crash_every == 0))
        if boundary:
            for f in pending:
                f.result()
            pending = []
            crash = args.crash_every and n % args.crash_every == 0
            before = chk.metrics(args.host, args.port).get("nvme_cache", {})  # {} with the tier off
            crash_rank = None
            if crash:
                world = before.get("world") or int(json.load(open(args.config)).get("world_size", 1))
                crash_rank = rng.randrange(world) if args.crash_rank == "random" else int(args.crash_rank)
                crashes += 1
            ok = restart(args.config, args.bin, log_lines, crash_rank=crash_rank)
            after = chk.metrics(args.host, args.port).get("nvme_cache", {})
            restarts += 1
            print(f"{'crash' if crash else 'restart'} {restarts}: {'; '.join(log_lines[-3 if crash else -2:])}; entries "
                  f"{before.get('entries')} -> {after.get('entries')}, adopted {after.get('adopted')}", flush=True)
            if not ok:
                findings.append({"restart": restarts, "what": "down/up did not complete cleanly"})
            # After a crash the world either adopted (every rank agreed) or
            # started fresh (no entry at all): never a partial index.
            if crash and after.get("entries", 0) != after.get("adopted", 0):
                findings.append({"restart": restarts, "what": "entries after the crash are not all adopted ones",
                                 "nvme_cache": after})
        d, q = rng.randrange(args.docs), rng.randrange(len(questions))
        body_extra = {}
        if rng.random() < 0.5:
            body_extra["prompt_cache_key"] = f"fuzz-doc-{d}"
        if rng.random() < 0.1:
            body_extra["prefix_cache"] = False
        cancel_after_ms = rng.randrange(20, 1500) if rng.random() * 100 < args.cancel_pct else None
        if args.clients > 1:
            pending.append(pool.submit(one, n, d, q, body_extra, cancel_after_ms))
            if len(pending) >= args.clients:
                pending[0].result()
                pending = [f for f in pending if not f.done()] + [f for f in pending if f.done() and f.exception()]
        else:
            one(n, d, q, body_extra, cancel_after_ms)
    for f in pending:
        f.result()
    pool.shutdown()
    steps.sort(key=lambda s: s["n"])
    m = chk.metrics(args.host, args.port)
    nv = m.get("nvme_cache", {})  # the driver also serves as a plain workload on a world without the tier
    tier_ok = nv.get("spill_failed", 0) == 0 and nv.get("restore_failed", 0) == 0 and nv.get("failures", 0) == 0 \
        and nv.get("pages_used", 0) <= nv.get("pages_total", 0)
    if not tier_ok:
        findings.append({"what": "tier failures or capacity breach", "nvme_cache": nv})
    repeats = sum(1 for s in steps if s.get("repeat"))
    cancelled = sum(1 for s in steps if "cancelled_after_ms" in s)
    # The service must have settled: nothing active, nothing queued, no
    # request stuck on a restore that never came.
    sc = m["scheduler"]
    if sc.get("active", 0) != 0 or sc.get("queued", 0) != 0 or m["service"].get("engine_failed"):
        findings.append({"what": "the service did not settle", "scheduler": {k: sc.get(k) for k in ("active", "queued")},
                         "engine_failed": m["service"].get("engine_failed")})
    print(f"\n{args.requests} requests, {repeats} repeats checked, {cancelled} cancelled, {restarts} restarts "
          f"({crashes} crashes), {len(findings)} finding(s)" +
          (f", {len(advisory)} transcript mismatch(es) advisory under {args.clients} clients" if advisory else ""))
    print("nvme_cache:", {k: nv.get(k) for k in ("entries", "spills", "spilled", "restores", "restored", "spill_failed",
                                                 "restore_failed", "evictions", "pinned", "adopted", "pages_used",
                                                 "pages_total")})
    print("prefix_cache:", {k: m["prefix_cache"].get(k) for k in ("hits", "misses", "evictions", "pinned")})
    for f in findings:
        print("FINDING:", json.dumps(f)[:400])
    print("PASS" if not findings else "FAIL")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "steps": steps, "findings": findings, "advisory": advisory,
                       "restarts": log_lines, "metrics": m, "ok": not findings}, fh, indent=1)
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
