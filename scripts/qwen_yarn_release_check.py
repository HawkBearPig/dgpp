#!/usr/bin/env python3
"""The opt-in YaRN release check: a real-checkpoint run near 256K and 512K.

This is a RELEASE CHECK, not a test. It is never part of CI and it refuses to
start without an explicit acknowledgement (`--run`, or DGPP_YARN_RELEASE_CHECK=1
in the environment) — before it opens a socket, so a stray call from a job, a
cron entry or a test glob prints why it stopped and measures nothing. What one
run costs: an idle two-Spark world serving the NVFP4 checkpoint with
deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json, and roughly
the hour the long prefills take. The procedure is
docs/qwen_yarn_release_check.md; the recipe it validates is
docs/qwen38_flash_next_plan.md §1.9.1.

Why it exists: every gate the repo carries proves the YaRN table is the table
vLLM builds (tests/unit/qwen_rope_scaling_test.cpp, tests/cuda/qsa_test.cu, the
oracle in tests/python/qwen_yarn_oracle.py) and that the memory plan does not
move (the plan check in tests/cuda/qwen_forward_test.cpp). None of them says
what the review asked for — that the engine, on a real checkpoint, actually
serves a 524 288-token request: a long prefill, an incremental decode, the
prefix cache, several streams at once — and still finds what it was told inside
the context. That is a cluster measurement, so it lives in a script.

For each length, in this order:
  * retrieval probes at several depths of one deterministic document, greedy,
    scored by whether the needle's digits come back. The first probe also
    measures the COLD PREFILL of the whole document, from the engine's own
    prefill_ms and computed-token counters plus the client's time to first
    token;
  * cache reuse: the first probe again, byte for byte, whose cached_tokens and
    prefill_ms say whether the shared prefix came from the arena or from a
    second prefill;
  * incremental decode: a long generation, paced from the stream's own arrival
    stamps;
  * concurrency: N streams over the same document at once (the TTFT
    distribution and the aggregate rate);
  * once per lane, a short-context control at the same seed. A YaRN regression
    shows up there first: an engine that scaled something it should not have
    lost pace or accuracy at 4 096 tokens, long before 512K is involved.

Every number also lands in a machine-readable JSON (--json-out), and the last
block is a human summary with the verdict. Peak memory is sampled from
nvidia-smi on the node the script runs on (rank 0) and the record names that
node; the peers' peaks come from scripts/node_probe.sh, which the runbook
quotes in the same table rather than pretending this script measured them.

Against a reference lane: `--reference-url` replays the SAME prompts, needles and
sampling against another OpenAI endpoint — the vLLM container with the same
checkpoint and the same YaRN settings (`YARN_ENABLE=true YARN_FACTOR=2.0
MAX_MODEL_LEN=524288`). Retrieval misses then attribute themselves: both lanes
miss → the model or the recipe, DGPP alone misses → a DGPP regression, DGPP
alone hits → put it in the record. The comparison is only honest if the two
ropes agree, so the check reads `/v1/models` on both sides and refuses a pair
whose advertised request limits differ unless `--allow-ceiling-mismatch`.

Usage:
  scripts/qwen_yarn_release_check.py --run [--host H --port P]
      [--lengths 262144 524288] [--depths 0.05 0.25 0.5 0.75 0.95]
      [--control-length 4096] [--decode-tokens 128] [--concurrency 4]
      [--seed 20260918] [--json-out FILE] [--reference-url http://HOST:PORT]
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_stream import read_completion  # noqa: E402
from serve_client import served_model  # noqa: E402
from site_env import default_host, http_port  # noqa: E402

# One record of the structured filler: flat, dense and deliberately unlike
# prose. The tokenizer's cost per record is stable, nothing in it can be
# guessed from the rest of the document, and its fields give the probes exact
# questions with one-token answers.
RECORD = ("Item {index:06d}: parcel {parcel} weighed {weight:.4f} kilograms in {city}, "
          "gate {gate}, shift {shift}, status {status}.\n")
NEEDLE = "IMPORTANT: the vault code for shipment {code_id} is {digits}. Remember it.\n"
CITIES = ("Arcadia", "Belvedere", "Calder", "Dunmore", "Eastvale", "Fenwick",
          "Glencarr", "Halloway", "Income", "Jamesport")
STATUSES = ("sealed", "in-transit", "held", "cleared", "rerouted", "quoted")
# The template's own overhead (the role markers and the trailing turn), folded
# into the calibration rather than assumed away from it.
TEMPLATE_OVERHEAD_TOKENS = 24


def ms(value, digits=0):
    """Milliseconds for the summary lines, with missing kept missing."""
    return "n/a" if value is None else f"{value:.{digits}f}"


class Lane:
    """One OpenAI endpoint: the DGPP world, or the reference stack beside it."""

    # The lengths this lane runs: the full set for DGPP, `--reference-lengths`
    # for the other one (the expensive lane is not always both).
    lengths = []

    def __init__(self, url, model=None, name="dgpp"):
        parsed = urlsplit(url if "://" in url else "http://" + url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError(f"not an http endpoint: {url}")
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.name, self.url = name, url
        self.model = model or served_model(self.host, self.port)
        self.surface = self.get("/v1/models")["data"][0]

    def get(self, path):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=60)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"GET {path} on {self.name}: HTTP {response.status}: "
                                   f"{body[:512]!r}")
            return json.loads(body)
        finally:
            conn.close()

    def metrics(self):
        return self.get("/v1/metrics")["scheduler"]

    def reported_limit(self):
        """The request context limit the model object advertises, if any.

        `context` and `rope_scaling` ride the DGPP build's /v1/models (added
        with the knob) and are the fastest witness that the ramp reached the
        process. A lane without them (vLLM, an older DGPP) reports None."""
        context = self.surface.get("context")
        return context.get("request_limit_tokens") if isinstance(context, dict) else None

    def ask(self, prompt, max_tokens, timeout=None):
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_tokens, "temperature": 0, "stream": True,
                           "stream_options": {"include_usage": True}})
        # A 512K prefill is minutes, not seconds; the budget scales with the
        # prompt so a hung endpoint still fails in hours rather than days.
        conn = http.client.HTTPConnection(
            self.host, self.port, timeout=timeout or (900 + len(prompt) // 2000))
        t0 = time.perf_counter()
        try:
            conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            result = read_completion(conn.getresponse())
        finally:
            conn.close()
        ended = time.perf_counter()
        return {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "usage": result["usage"], "finish": result["finish"], "text": result["text"],
                "wall_ms": 1000 * (ended - t0),
                "ttft_ms": 1000 * (result["first"] - t0) if result["first"] is not None else None,
                "last_ms": 1000 * (result["last"] - t0) if result["last"] is not None else None,
                "tokens": result["tokens"]}


def filler(seed, records):
    """The deterministic filler lines: the same seed, the same document."""
    rng = random.Random(seed)
    return [RECORD.format(index=i, parcel=f"PX-{rng.randrange(1 << 20):07d}",
                          weight=rng.uniform(0.1, 40.0), city=rng.choice(CITIES),
                          gate=rng.randrange(1, 24), shift=rng.choice("ABC"),
                          status=rng.choice(STATUSES))
            for i in range(records)]


def calibrate(lane, seed, records=256):
    """Tokens per filler record, measured on this endpoint with this template."""
    prompt = "".join(filler(seed, records))
    answer = lane.ask(prompt, 1)
    used = answer["usage"]["prompt_tokens"]
    per_record = (used - TEMPLATE_OVERHEAD_TOKENS) / records
    if not math.isfinite(per_record) or per_record <= 0:
        raise RuntimeError(f"{lane.name}: the token calibration is not positive")
    return {"probe_records": records, "probe_prompt_tokens": used, "tokens_per_record": per_record}


def plant(lines, depths, seed):
    """Splice one needle per depth into a copy of the filler; return the probes.

    A needle's id and code are a function of the seed, so a rerun months later
    asks the same question about the same position."""
    rng = random.Random(seed * 31 + 7)
    document, probes = list(lines), []
    for depth in depths:
        at = min(len(document) - 1, max(1, int(round(depth * len(document)))))
        code_id = f"VL-{rng.randrange(10000, 99999)}"
        digits = f"{rng.randrange(1000, 9999)}{rng.randrange(1000, 9999)}"
        document.insert(at, NEEDLE.format(code_id=code_id, digits=digits))
        probes.append({"depth": depth, "line": at, "code_id": code_id, "digits": digits,
                       "question": f"What is the vault code for shipment {code_id}? "
                                   f"Answer with the digits only, nothing else."})
    return "".join(document), probes


def wait_idle(lane, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        metrics = lane.metrics()
        if not (metrics["active"] or metrics["queued"] or metrics.get("prefilling")):
            return metrics
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{lane.name}: the scheduler never went idle")
        time.sleep(0.25)


class MemorySampler(threading.Thread):
    """nvidia-smi on this node, so the record has a peak and not a wish."""

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval, self.stop = interval, threading.Event()
        self.samples = []
        self.present = shutil.which("nvidia-smi") is not None

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, timeout=15, check=True).stdout
            except Exception:                                                # noqa: BLE001
                return
            for line in out.splitlines():
                if line.strip().isdigit():
                    self.samples.append(int(line.strip()))
            self.stop.wait(self.interval)

    def report(self):
        if not self.samples:
            return {"available": self.present, "node": os.uname().nodename, "peak_mib": None,
                    "note": "no nvidia-smi samples on this node; take every rank's with "
                            "scripts/node_probe.sh and paste them into the record"}
        return {"available": True, "node": os.uname().nodename, "peak_mib": max(self.samples),
                "samples": len(self.samples)}


def run_probe(lane, document, probe, max_tokens, cold):
    """One retrieval probe, with the engine's own counters around it."""
    prompt = document + "\n" + probe["question"]
    before = lane.metrics()
    answer = lane.ask(prompt, max_tokens)
    after = lane.metrics()
    usage = answer["usage"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    computed = after["prompt_tokens_computed"] - before["prompt_tokens_computed"]
    prefill_ms = after["prefill_ms"] - before["prefill_ms"]
    hit = probe["digits"] in re.sub(r"[^0-9]", "", answer["text"])
    return {"depth": probe["depth"], "expected": probe["digits"], "answer": answer["text"][:160],
            "hit": hit, "prompt_tokens": usage["prompt_tokens"], "cached_tokens": cached,
            "computed_tokens": computed, "prefill_ms": prefill_ms,
            "prefill_ms_per_token": prefill_ms / computed if computed else None,
            "ttft_ms": answer["ttft_ms"], "wall_ms": answer["wall_ms"], "finish": answer["finish"],
            "cold": cold, "prompt_sha256": answer["prompt_sha256"]}


def retrieval_phase(lane, name, document, probes, max_tokens, into):
    results = [run_probe(lane, document, probe, max_tokens, cold=index == 0)
               for index, probe in enumerate(probes)]
    for out in results:
        print(f"[{lane.name}/{name}] depth {out['depth']:.2f} prompt {out['prompt_tokens']} tok "
              f"(cached {out['cached_tokens']}) prefill {out['prefill_ms']:.0f} ms "
              f"ttft {out['ttft_ms']:.0f} ms -> {'HIT' if out['hit'] else 'MISS'} "
              f"(want {out['expected']}, got {out['answer']!r})", flush=True)
    hits = sum(1 for out in results if out["hit"])
    into[name] = {"probes": results, "hits": hits, "requests": len(results),
                  "hit_rate": hits / len(results) if results else None,
                  "cold_prefill_ms": results[0]["prefill_ms"] if results else None,
                  "cold_prefill_ms_per_token": results[0]["prefill_ms_per_token"] if results else None}
    return into[name]


def concurrent_phase(lane, prompts, max_tokens):
    """The concurrency shape: one wall, several streams, no shared client."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        started = time.perf_counter()
        futures = [pool.submit(lane.ask, prompt, max_tokens) for prompt in prompts]
        answers, errors = [], []
        for future in concurrent.futures.as_completed(futures):
            try:
                answers.append(future.result())
            except Exception as error:                                       # noqa: BLE001
                errors.append(f"{type(error).__name__}: {error}")
        elapsed = time.perf_counter() - started
    ttft = sorted(out["ttft_ms"] for out in answers if out["ttft_ms"] is not None)
    per_token = [1000 * (out["last_ms"] - out["ttft_ms"]) / max(1, out["tokens"] - 1)
                 for out in answers if out["last_ms"] is not None and out["ttft_ms"] is not None]
    return {"requests": len(prompts), "completed": len(answers), "errors": errors, "wall_s": elapsed,
            "completion_tokens": sum(out["tokens"] for out in answers),
            "aggregate_tokens_per_s": sum(out["tokens"] for out in answers) / elapsed
            if elapsed > 0 else None,
            "ttft_ms": ttft, "ttft_median_ms": statistics.median(ttft) if ttft else None,
            "per_token_ms": per_token}


def run_lane(lane, args, record):
    lane.calibration = calibrate(lane, args.seed)
    print(f"[{lane.name}/calibrate] {lane.calibration['probe_records']} records -> "
          f"{lane.calibration['probe_prompt_tokens']} prompt tokens "
          f"({lane.calibration['tokens_per_record']:.3f} tok/record)", flush=True)
    out = record["results"][lane.name] = {"calibration": lane.calibration, "lengths": {}}
    per_record = lane.calibration["tokens_per_record"]
    for length in lane.lengths:
        slot = out["lengths"][str(length)] = {}
        lines = filler(args.seed, max(64, int(length // per_record)))
        document, probes = plant(lines, args.depths, args.seed)
        print(f"[{lane.name}/{length}] {len(lines)} records, {len(document)} chars, "
              f"{len(probes)} needles at depths {args.depths}", flush=True)
        retrieval_phase(lane, "retrieval", document, probes, args.probe_tokens, slot)

        # Cache reuse: the first probe again, byte for byte. The document is a
        # shared prefix, so anything but a full prefill here came from the
        # arena — and a cached_ratio near zero means prefix_cache_gib is 0.
        reuse = run_probe(lane, document, probes[0], args.probe_tokens, cold=False)
        reuse["cached_ratio"] = reuse["cached_tokens"] / reuse["prompt_tokens"] \
            if reuse["prompt_tokens"] else None
        slot["cache_reuse"] = reuse
        print(f"[{lane.name}/{length}] cache reuse: cached {reuse['cached_tokens']}/"
              f"{reuse['prompt_tokens']} tokens ({100 * (reuse['cached_ratio'] or 0):.1f}%) "
              f"in {reuse['prefill_ms']:.0f} ms", flush=True)

        # Incremental decode: a long generation, paced from the stream.
        decode_doc = "".join(filler(args.seed + 1, max(64, int(args.decode_context // per_record))))
        answer = lane.ask(decode_doc + "\nSummarize every item above in as many sentences as you "
                                      "can.", args.decode_tokens)
        pace = 1000 * (answer["last_ms"] - answer["ttft_ms"]) / max(1, answer["tokens"] - 1) \
            if answer["last_ms"] is not None and answer["ttft_ms"] is not None else None
        slot["decode"] = {"prompt_tokens": answer["usage"]["prompt_tokens"],
                          "completion_tokens": answer["tokens"], "ttft_ms": answer["ttft_ms"],
                          "wall_ms": answer["wall_ms"], "ms_per_token": pace,
                          "finish": answer["finish"],
                          "text_sha256": hashlib.sha256(answer["text"].encode()).hexdigest()}
        print(f"[{lane.name}/{length}] decode: {answer['tokens']} tokens, "
              f"ttft {ms(answer['ttft_ms'])} ms, decode {ms(pace, 2)} ms/token", flush=True)

        # Concurrency: the same document, several streams at once.
        slot["concurrent"] = concurrent_phase(
            lane, [document + "\n" + probe["question"] for probe in probes[:args.concurrency]],
            args.probe_tokens)
        print(f"[{lane.name}/{length}] concurrent: {slot['concurrent']['completed']}/"
              f"{slot['concurrent']['requests']} streams in {slot['concurrent']['wall_s']:.1f} s, "
              f"{slot['concurrent']['aggregate_tokens_per_s']:.2f} tok/s aggregate", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true",
                        help="acknowledge that this is a long, real-checkpoint release check and "
                             "not a CI test; DGPP_YARN_RELEASE_CHECK=1 says the same")
    parser.add_argument("--host", help="default: the selected deployment's head node")
    parser.add_argument("--port", type=int, help="default: the selected deployment's HTTP port")
    parser.add_argument("--model", help="override the endpoint's advertised model id")
    parser.add_argument("--lengths", type=int, nargs="+", default=[262144, 524288],
                        help="document sizes: near the native ceiling and the YaRN one")
    parser.add_argument("--depths", type=float, nargs="+", default=[0.05, 0.25, 0.50, 0.75, 0.95],
                        help="relative positions the needles are planted at")
    parser.add_argument("--control-length", type=int, default=4096,
                        help="the short-context control (a YaRN regression shows here first)")
    parser.add_argument("--probe-tokens", type=int, default=24)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--decode-context", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--reference-url", help="the vLLM lane as http://HOST:PORT")
    parser.add_argument("--reference-model", help="the reference lane's model id")
    parser.add_argument("--reference-lengths", type=int, nargs="+", default=None,
                        help="lengths to replay on the reference lane (default: --lengths)")
    parser.add_argument("--expect-limit", type=int, default=524288,
                        help="the request context limit the DGPP endpoint must advertise")
    parser.add_argument("--allow-ceiling-mismatch", action="store_true",
                        help="compare lanes whose advertised request limits differ")
    parser.add_argument("--min-hit-rate", type=float, default=0.6,
                        help="the absolute retrieval floor when no reference lane ran")
    parser.add_argument("--max-shortfall", type=int, default=0,
                        help="with a reference lane: how many probes DGPP may hit fewer of")
    parser.add_argument("--mem-sample-s", type=float, default=2.0)
    parser.add_argument("--json-out", help="where the machine-readable record goes")
    parser.add_argument("--tag", default="", help="a name for this run, carried into the record")
    args = parser.parse_args()

    if not args.run and os.environ.get("DGPP_YARN_RELEASE_CHECK") != "1":
        parser.error("this is the opt-in release check, not a test: it runs a real checkpoint for "
                     "roughly an hour on an idle two-Spark world. Read "
                     "docs/qwen_yarn_release_check.md, then pass --run (or set "
                     "DGPP_YARN_RELEASE_CHECK=1).")
    if not args.lengths or any(n < 1024 for n in args.lengths):
        parser.error("--lengths must be non-empty and at least 1024 tokens each")
    if not args.depths or any(not 0.0 < d < 1.0 for d in args.depths):
        parser.error("--depths must be inside (0, 1)")
    if args.concurrency < 1 or args.decode_tokens < 2 or args.probe_tokens < 2:
        parser.error("--concurrency, --decode-tokens and --probe-tokens must fit their probes")
    if args.mem_sample_s <= 0:
        parser.error("--mem-sample-s must be positive")

    host = args.host or default_host()
    port = args.port or http_port()
    lanes = [Lane(f"{host}:{port}", args.model, "dgpp")]
    if args.reference_url:
        lanes.append(Lane(args.reference_url, args.reference_model, "reference"))
    dgpp = lanes[0]
    reported = dgpp.reported_limit()
    if reported is not None and reported < max(args.lengths) and not args.allow_ceiling_mismatch:
        raise SystemExit(f"the DGPP endpoint advertises a {reported}-token request limit and the "
                         f"run asks for {max(args.lengths)}: engine.rope_scaling did not reach the "
                         f"process (deploy/README.md, the YaRN template). Refusing to write a "
                         f"record that would measure truncation.")
    if reported is not None and reported < args.expect_limit:
        raise SystemExit(f"the DGPP endpoint advertises a {reported}-token request limit, below "
                         f"the {args.expect_limit} the run expects (--expect-limit): the verdict "
                         f"would measure a different ceiling than the record's tag claims.")
    if len(lanes) == 2 and not args.allow_ceiling_mismatch:
        other = lanes[1].reported_limit()
        if reported is not None and other is not None and reported != other:
            raise SystemExit(f"the lanes' request limits disagree (dgpp {reported}, reference "
                             f"{other}): the retrieval diff would compare different contexts. "
                             f"Match the recipes or pass --allow-ceiling-mismatch.")

    record = {"schema_version": 1, "kind": "qwen_yarn_release_check", "tag": args.tag,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": args.seed,
              "args": {key: value for key, value in vars(args).items() if key != "run"},
              "lanes": {lane.name: {"url": lane.url, "model": lane.model,
                                    "reported_limit": lane.reported_limit(),
                                    "rope_scaling": lane.surface.get("rope_scaling"),
                                    "context": lane.surface.get("context")} for lane in lanes},
              "results": {}, "peak_memory": None}
    sampler = MemorySampler(args.mem_sample_s)
    sampler.start()
    print("[lanes] " + ", ".join(f"{lane.name}={lane.url} model={lane.model}" for lane in lanes),
          flush=True)
    reference_lengths = args.reference_lengths or args.lengths
    for lane in lanes:
        wait_idle(lane, timeout=120)
        lane.lengths = reference_lengths if lane is not dgpp else args.lengths
        run_lane(lane, args, record)
        # The short-context control, per lane: the same probes, no long context.
        lines = filler(args.seed, max(64, int(args.control_length //
                                             lane.calibration["tokens_per_record"])))
        document, probes = plant(lines, args.depths, args.seed)
        retrieval_phase(lane, "short_context", document, probes, args.probe_tokens,
                        record["results"][lane.name])
        wait_idle(lane, timeout=120)
    sampler.stop.set()
    time.sleep(0.1)
    record["peak_memory"] = sampler.report()

    # The verdict: retrieval first (against the reference when one ran, the
    # absolute floor otherwise), then the short-context control, then the
    # shapes of the timings, which are reported and not gated — a release
    # check that guesses its own thresholds has no business saying PASS.
    checks = []
    for length in args.lengths:
        mine = record["results"]["dgpp"]["lengths"][str(length)]["retrieval"]
        line = {"check": f"retrieval@{length}", "dgpp_hits": mine["hits"],
                "requests": mine["requests"]}
        if len(lanes) == 2 and str(length) in record["results"].get("reference", {}).get("lengths", {}):
            theirs = record["results"]["reference"]["lengths"][str(length)]["retrieval"]
            line.update({"reference_hits": theirs["hits"],
                         "ok": mine["hits"] >= theirs["hits"] - args.max_shortfall})
        else:
            line.update({"floor": args.min_hit_rate, "ok": mine["hit_rate"] >= args.min_hit_rate})
        checks.append(line)
    control = record["results"]["dgpp"].get("short_context", {})
    checks.append({"check": f"short_context@{args.control_length}",
                   "dgpp_hits": control.get("hits", 0), "requests": control.get("requests", 0),
                   "floor": args.min_hit_rate,
                   "ok": (control.get("hit_rate") or 0) >= args.min_hit_rate})
    verdict = {"checks": checks, "pass": all(line.get("ok", False) for line in checks)}
    record["verdict"] = verdict
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    print(f"\n== YaRN release check {'PASS' if verdict['pass'] else 'FAIL'} ==", flush=True)
    for line in checks:
        print("  " + " ".join(f"{key}={value}" for key, value in line.items()), flush=True)
    for name, result in record["results"].items():
        for length, slot in result["lengths"].items():
            print(f"  {name} @{length}: cold prefill "
                  f"{ms(slot['retrieval']['cold_prefill_ms'])} ms ("
                  f"{ms(slot['retrieval']['cold_prefill_ms_per_token'], 4)} ms/token), decode "
                  f"{ms(slot['decode']['ms_per_token'], 2)} ms/token, reuse "
                  f"{100 * (slot['cache_reuse'].get('cached_ratio') or 0):.1f}%, concurrent "
                  f"{slot['concurrent']['aggregate_tokens_per_s']:.2f} tok/s", flush=True)
        control_slot = result.get("short_context", {})
        print(f"  {name} control @{args.control_length}: "
              f"{control_slot.get('hits', 0)}/{control_slot.get('requests', 0)} needles", flush=True)
    memory = record["peak_memory"]
    print("  peak memory: " + (f"{memory['peak_mib']} MiB on {memory['node']}"
                               if memory.get("peak_mib") else memory.get("note", "not sampled")),
          flush=True)
    if args.json_out:
        print(f"  record: {args.json_out}", flush=True)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
