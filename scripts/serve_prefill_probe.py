#!/usr/bin/env python3
"""Measure cold prefill through the serving endpoint on an otherwise idle server.

LEN is an approximate token count, calibrated from prepared GSM8K prose.
Each request has a distinct prefix; JSON records actual prompt/computed/cache
counts, engine prefill time and visible client TTFT. Use the same --tag and
--seed in matched fresh-server runs. --no-think fixes the template's mode.

Usage: serve_prefill_probe.py HOST PORT LEN... [--repeat N] [--json-out FILE]
"""
import argparse
import hashlib
import http.client
import json
import math
import random
import statistics
import time
from pathlib import Path

from data_paths import data_dir, require_file
from bench_stream import read_completion


def get(host, port, path, *, timeout=60):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"GET {path}: HTTP {response.status}: {body[:1024]!r}")
        return json.loads(body)
    finally:
        conn.close()


def prom_value(text, name):
    best = None
    for line in text.splitlines():
        if line.startswith(name + "{") or line == name + " " or line.startswith(name + " "):
            try:
                value = float(line.rsplit(" ", 1)[1])
            except ValueError:
                continue
            if best is None or value > best:
                best = value
    return best


def probe_metrics(host, port):
    """Return ("dgpp", scheduler) or ("vllm", prometheus_text) for the server."""
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", "/metrics")
        response = conn.getresponse()
        body = response.read().decode()
        if response.status == 200:
            try:
                return "dgpp", json.loads(body)["scheduler"]
            except (ValueError, KeyError):
                return "vllm", body
        raise RuntimeError(f"GET /metrics: HTTP {response.status}: {body[:1024]!r}")
    finally:
        conn.close()


def idle_metrics(host, port, timeout=0, completed_after=None):
    deadline = time.monotonic() + timeout
    while True:
        kind, metrics = probe_metrics(host, port)
        if kind == "dgpp":
            idle = not metrics["active"] and not metrics["queued"]
            done = completed_after is None or metrics["prompts_prefilled"] > completed_after
        else:
            idle = prom_value(metrics, "vllm:num_requests_running") == 0 and \
                prom_value(metrics, "vllm:num_requests_waiting") == 0
            done = True
        if idle and done:
            return metrics if kind == "dgpp" else None
        if time.monotonic() >= deadline:
            raise RuntimeError("prefill probe requires an otherwise idle server")
        # The SSE finish can arrive before the service publishes the retired
        # scheduler snapshot. Wait for that snapshot only after our request;
        # measured_prefill still rejects work from any intervening request.
        time.sleep(0.01)


def prompt_of(words, n_words, nonce, rng):
    if n_words > len(words):
        raise ValueError(f"prompt needs {n_words} words but dataset contains only {len(words)}")
    start = rng.randrange(0, max(1, len(words) - n_words - 1))
    return f"Reference {nonce}. Summarize the following in one sentence.\n\n" + \
        " ".join(words[start:start + n_words])


def ask(host, port, model, prompt, no_think=False):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 1, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    conn = http.client.HTTPConnection(host, port, timeout=1200)
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(payload),
                     {"Content-Type": "application/json"})
        result = read_completion(conn.getresponse())
        ended = time.perf_counter()
    finally:
        conn.close()
    # A one-token completion may contain only an invisible special token.
    # Keep missing TTFT explicit; a role or finish event is not visible output.
    return {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "usage": result["usage"], "finish": result["finish"], "text": result["text"],
            "wall_ms": 1000 * (ended - t0),
            "ttft_ms": 1000 * (result["first"] - t0) if result["first"] is not None else None}


def measured_prefill(before, after, response):
    usage = response["usage"]
    prompt = usage["prompt_tokens"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if after is None:
        # Prometheus-only server (vLLM): on an idle server, client TTFT is the
        # prefill time up to first-token scheduling latency.
        if response["ttft_ms"] is None:
            raise RuntimeError("no visible first token; cannot time prefill without scheduler metrics")
        return {**response, "prompt_tokens": prompt, "cached_tokens": cached,
                "computed_tokens": prompt - cached, "prefill_ms": response["ttft_ms"],
                "prefill_ms_per_token": response["ttft_ms"] / (prompt - cached),
                "prefill_source": "client_ttft"}
    computed = after["prompt_tokens_computed"] - before["prompt_tokens_computed"]
    milliseconds = after["prefill_ms"] - before["prefill_ms"]
    if after["prompts_prefilled"] - before["prompts_prefilled"] != 1 or computed != prompt - cached:
        raise RuntimeError("prefill metrics do not describe exactly this request; check concurrent traffic")
    if cached:
        raise RuntimeError(f"cold prefill probe attached {cached} tokens; use a new tag or fresh server")
    if computed <= 0 or not math.isfinite(milliseconds) or milliseconds <= 0:
        raise RuntimeError("prefill metrics did not report positive computed work")
    return {**response, "prompt_tokens": prompt, "cached_tokens": cached,
            "computed_tokens": computed, "prefill_ms": milliseconds,
            "prefill_ms_per_token": milliseconds / computed, "prefill_source": "engine"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("lengths", type=int, nargs="*", default=[512, 2048, 8192])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--data", help="GSM8K JSONL file used as prompt text")
    parser.add_argument("--tag", default="", help="cold-prefix identifier, identical in matched fresh-server runs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--interval", type=float, default=0,
                        help="seconds between requests; use 1 to isolate requests in an nsys burst profile")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.repeat < 1 or not args.lengths or any(n < 1 for n in args.lengths):
        parser.error("repeat count and prompt lengths must be positive")
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("interval must be finite and nonnegative")
    data_path = require_file(args.data or data_dir() / "gsm8k_test.jsonl")
    model = args.model or get(args.host, args.port, "/v1/models")["data"][0]["id"]
    words = []
    raw = data_path.read_bytes()
    for line in raw.splitlines():
        words.extend(json.loads(line)["question"].split())
    rng = random.Random(args.seed)
    def nonce(value):
        return f"{args.tag}-{value}" if args.tag else value
    initial = idle_metrics(args.host, args.port)
    calibration = ask(args.host, args.port, model, prompt_of(words, 400, nonce("cal-0"), rng), args.no_think)
    idle_metrics(args.host, args.port, timeout=5,
                 completed_after=initial["prompts_prefilled"] if initial else None)
    tokens = calibration["usage"]["prompt_tokens"]
    per_word = (tokens - 30) / 400
    if per_word <= 0:
        raise RuntimeError("invalid token-to-word calibration")
    print(f"[calibrate] 400 words -> {tokens} prompt tokens ({per_word:.3f} tok/word)", flush=True)
    report = {"schema_version": 1, "model": model, "tag": args.tag, "seed": args.seed,
              "thinking": False if args.no_think else "server_default", "interval_s": args.interval,
              "data_sha256": hashlib.sha256(raw).hexdigest(), "calibration": calibration, "samples": []}
    for length in args.lengths:
        n_words = max(8, int((length - 30) / per_word))
        samples = []
        for repeat in range(args.repeat):
            if args.interval:
                time.sleep(args.interval)
            prompt = prompt_of(words, n_words, nonce(f"n{length}-r{repeat}-{rng.randrange(1 << 30)}"), rng)
            before = idle_metrics(args.host, args.port)
            response = ask(args.host, args.port, model, prompt, args.no_think)
            after = idle_metrics(args.host, args.port, timeout=5,
                                 completed_after=before["prompts_prefilled"] if before else None)
            sample = measured_prefill(before, after, response)
            sample.update(requested_tokens=length, repeat=repeat)
            samples.append(sample)
            report["samples"].append(sample)
            if args.json_out:
                Path(args.json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            ttft = f"{sample['ttft_ms']:.0f}" if sample["ttft_ms"] is not None else "missing"
            print(f"[n{length} r{repeat}] prompt {sample['prompt_tokens']} tok, computed {sample['computed_tokens']} "
                  f"in {sample['prefill_ms']:.0f} ms = {sample['prefill_ms_per_token']:.2f} ms/token; "
                  f"ttft {ttft} ms", flush=True)
        print(f"[n{length}] median: {statistics.median(s['prefill_ms_per_token'] for s in samples):.2f} ms/token; "
              f"prefill {statistics.median(s['prefill_ms'] for s in samples):.0f} ms; "
              f"wall {statistics.median(s['wall_ms'] for s in samples):.0f} ms", flush=True)


if __name__ == "__main__":
    main()
