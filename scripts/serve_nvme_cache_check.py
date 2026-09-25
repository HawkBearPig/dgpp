#!/usr/bin/env python3
"""The NVMe cold tier's fabric check (issue #26): a long document's entries
spill, other documents push them out of the memory arena, and the document
comes back from disk for a later question — restored, attached, and
answered with the same tokens the in-memory hit produced.

Boot the world with a small arena so eviction is quick and the tier on, e.g.

  scripts/dgpp-cluster up --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json \\
      --knobs '--nvme-cache-path ~/dgpp/nvme-cache --nvme-cache-gib 32 --prefix-cache-gib 0.15'

then run this against it. Each request is greedy and streamed; TTFT is the
first delta's arrival. The steps:

  1. document A + question 0      cold prefill; A's entries spill in the background
  2. document A + question 1      the changed question attaches in memory (the body cut)
  3. K other documents            their entries evict A's from the arena (the copies stay on disk)
  4. document A + question 1      the memory miss restores A from disk, then attaches
  5. document A + question 0      the restored entry serves the original question

Reports every step's TTFT and prompt/cached tokens, the transcript equality
of steps 2 and 4, and the nvme_cache / prefix_cache metric deltas. Exit 0
when a restore was committed and the transcripts match; JSON to --out.
"""
import argparse
import http.client
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_yarn_release_check import filler  # noqa: E402
from serve_client import served_model  # noqa: E402

CACHE_KEYS = ("spills", "spilled", "spill_failed", "spill_skipped", "restores", "restored",
              "restore_failed", "tokens_restored", "blocks_shared", "evictions", "entries",
              "pages_used", "pages_total", "bytes_written", "bytes_read", "failures")
PREFIX_KEYS = ("hits", "misses", "tokens_saved", "evictions", "entries", "snapshots", "close_entries")


def metrics(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=30)
    try:
        conn.request("GET", "/v1/metrics")
        return json.loads(conn.getresponse().read())
    finally:
        conn.close()


def ask(host, port, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"clear_thinking": False}}
    conn = http.client.HTTPConnection(host, port, timeout=3600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:400]!r}")
    ttft, text, usage = None, [], {}
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                try:
                    obj = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {})
                    piece = d.get("reasoning_content") or d.get("content")
                    if piece:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        text.append(piece)
    conn.close()
    total = (time.perf_counter() - t0) * 1000
    return {"ttft_ms": ttft, "total_ms": total, "text": "".join(text), "usage": usage}


def wait_for(host, port, key, at_least, timeout_s=120):
    """Polls the nvme_cache block until `key` reaches `at_least`; returns the block."""
    deadline = time.time() + timeout_s
    while True:
        m = metrics(host, port)["nvme_cache"]
        if m.get(key, 0) >= at_least:
            return m
        if time.time() > deadline:
            raise RuntimeError(f"nvme_cache.{key} never reached {at_least} (at {m.get(key)})")
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--records", type=int, default=700, help="filler records per document (~45 tokens each)")
    ap.add_argument("--others", type=int, default=3, help="other documents that push A out of the arena")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()
    if args.host is None or args.port is None:
        from site_env import default_host, http_port  # noqa: E402
        args.host = args.host or default_host()
        args.port = args.port or http_port()
    model = args.model or served_model(args.host, args.port)

    before = metrics(args.host, args.port)
    if not before.get("nvme_cache", {}).get("enabled"):
        print("the NVMe cache is not enabled on this world (nvme_cache.enabled is false)", file=sys.stderr)
        return 2
    doc = lambda seed: "".join(filler(seed, args.records))  # noqa: E731
    question = lambda i: f"\nQuestion {i}: write a numbered list describing the first {30 + i} parcels and their cities."  # noqa: E731
    doc_a = doc(args.seed)
    steps = []

    def step(name, prompt, expect=None):
        r = ask(args.host, args.port, model, prompt, args.max_tokens)
        r["step"] = name
        r["cached_tokens"] = r["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
        r["prompt_tokens"] = r["usage"].get("prompt_tokens")
        steps.append(r)
        print(f"{name:<44} ttft {r['ttft_ms']:8.0f} ms  prompt {r['prompt_tokens']}  cached {r['cached_tokens']}")
        return r

    step("1 document A + Q0 (cold)", doc_a + question(0))
    spilled = wait_for(args.host, args.port, "spilled", 1)["spilled"]
    print(f"   spilled so far: {spilled}")
    hit = step("2 document A + Q1 (memory hit)", doc_a + question(1))
    for k in range(args.others):
        step(f"3.{k + 1} document B{k + 1} + Q0 (cold, evicting A)", doc(args.seed + 1000 * (k + 1)) + question(0))
    pc = metrics(args.host, args.port)["prefix_cache"]
    print(f"   arena: {pc['entries']}/{pc['slots']} entries, {pc['evictions']} evictions so far")
    restored_before = metrics(args.host, args.port)["nvme_cache"]["restored"]
    back = step("4 document A + Q1 (restore from disk)", doc_a + question(1))
    step("5 document A + Q0 (memory hit after the restore)", doc_a + question(0))
    after = metrics(args.host, args.port)
    nv = after["nvme_cache"]
    restored = nv["restored"] - restored_before
    same = hit["text"] == back["text"]
    print()
    print("nvme_cache:", {k: nv.get(k) for k in CACHE_KEYS})
    print("prefix_cache deltas:", {k: after["prefix_cache"].get(k, 0) - before["prefix_cache"].get(k, 0)
                                   for k in PREFIX_KEYS})
    print(f"restores committed during step 4: {restored}; step 2 and step 4 transcripts "
          f"{'IDENTICAL' if same else 'DIFFER'}; restore failures {nv['restore_failed']}, "
          f"spill failures {nv['spill_failed']}, skipped {nv['spill_skipped']}")
    ok = restored >= 1 and same and back["cached_tokens"] and back["cached_tokens"] > 0
    report = {"model": model, "records": args.records, "steps": [{k: v for k, v in s.items() if k != "text"}
                                                                  for s in steps],
              "transcripts_identical": same, "before": before, "after": after, "ok": bool(ok)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
