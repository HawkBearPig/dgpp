#!/usr/bin/env python3
"""Measure decode throughput at fixed concurrency against dgpp-serve.

Each phase sends concurrent streaming requests with distinct long-answer
prompts. Requests are greedy with thinking disabled unless overridden.
The report includes usage token counts, time to first visible output and
client update pace, plus request-wall throughput and the explicitly named
legacy output-span rate. SSE updates can carry several tokens.

Use --classes to run separate sweeps for prose, code, JSON, math and chat.
Use --temperature to measure sampled requests.

    serve_load.py HOST PORT [--concurrency 1,2,4] [--max-tokens 320] [--think]
                            [--classes prose,code,json,math,chat] [--temperature T]
"""


import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import statistics
from datetime import datetime
from bench_stream import read_completion, phase_metrics

PROMPTS = [
    "Write a detailed technical essay of at least 800 words on how a paged key-value cache works in a "
    "transformer inference engine, covering block tables, eviction, prefix sharing and the trade-offs "
    "against a contiguous cache. Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on speculative decoding with a draft model: "
    "the verify pass, acceptance rules, why throughput rises, when it falls, and the memory bandwidth "
    "argument. Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on tensor parallelism across several GPUs "
    "connected by RDMA: sharding the weights, the all-reduce at each block boundary, the latency budget "
    "per step and the failure modes. Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on 4-bit weight quantization for large "
    "mixture-of-experts models: block scales, the accuracy cost, the kernels needed and how to validate "
    "the result. Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on CUDA graphs for decode steps: capture, "
    "replay, what they save, the constraints on the kernels inside them and how to debug a bad capture. "
    "Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on the RoCE fabric of a small GPU cluster: "
    "queue pairs, incast drops, pacing, and how to measure whether the network or the GPU bounds a step. "
    "Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on the request scheduler of an inference "
    "server: admission, the KV budget, round-robin decode, cancellation and fairness under load. Use "
    "section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on tokenizers for large language models: "
    "byte-level BPE, pre-tokenization, special tokens, chat templates and the pitfalls of a mismatch "
    "between training and serving. Use section headings and full paragraphs.",
]

# The per-class corpus. Prompt 0 of every class is the same TEXT as that class in
# scripts/fabric_mtp_classes.sh, which has been the project's class corpus since
# 2026-09-05 and produced every published per-class table — so a concurrency-1
# phase here is comparable to those numbers rather than to a fresh set of words.
# Prompts 1..3 are of the same kind and exist only because a phase at
# concurrency c needs c distinct prompts: repeating one would share a decode
# path and attach to the prefix cache, and neither is what a class costs under
# load. check_corpus() below re-reads the shell script and fails if prompt 0 has
# drifted from it.
CLASSES = {
    "prose": [
        "Write a long, detailed history of the Roman Republic from its founding to the rise of "
        "Augustus, one era per paragraph.",
        "Write a long, detailed history of the Hanseatic League from its founding to its dissolution, "
        "one era per paragraph.",
        "Write a long, detailed history of the silk trade in Lyon from its arrival to its decline, "
        "one era per paragraph.",
        "Write a long, detailed history of the Dutch water boards from their origin to the present, "
        "one era per paragraph.",
    ],
    "code": [
        "Write a Python module that parses an OpenAI-style SSE stream of chat completion chunks into "
        "a single message object, with type hints, docstrings, and a small set of unit tests using "
        "pytest.",
        "Write a Python module implementing an LRU cache with a byte budget and per-entry sizes, with "
        "type hints, docstrings, and a small set of unit tests using pytest.",
        "Write a Python module that reads a safetensors header and reports each tensor's name, dtype "
        "and shape, with type hints, docstrings, and a small set of unit tests using pytest.",
        "Write a Python module implementing a token bucket rate limiter usable from several threads, "
        "with type hints, docstrings, and a small set of unit tests using pytest.",
    ],
    "json": [
        "Return a JSON array of 25 objects, each with the fields country, capital, "
        "population_millions and currency, for 25 different countries. Output only the JSON.",
        "Return a JSON array of 25 objects, each with the fields city, country, population_millions "
        "and founded_year, for 25 different cities. Output only the JSON.",
        "Return a JSON array of 25 objects, each with the fields element, symbol, atomic_number and "
        "group, for 25 different chemical elements. Output only the JSON.",
        "Return a JSON array of 25 objects, each with the fields river, continent, length_km and "
        "mouth, for 25 different rivers. Output only the JSON.",
    ],
    "math": [
        "A train leaves city A at 60 km/h and another leaves city B, 450 km away, at 90 km/h toward "
        "it 30 minutes later. Work out step by step when and where they meet, then generalize the "
        "formula and check it with two other examples.",
        "A cistern is filled by two pipes in 12 and 18 minutes and emptied by a third in 24. Work out "
        "step by step how long it takes to fill, then generalize the formula and check it with two "
        "other examples.",
        "Find the area between the curves y = x^2 and y = 2x + 3. Work it out step by step, then "
        "generalize the method and check it with two other pairs of curves.",
        "A bag holds 7 red and 5 blue balls and three are drawn without replacement. Work out step by "
        "step the probability of exactly two red, then generalize the formula and check it with two "
        "other bags.",
    ],
    "chat": [
        "Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same "
        "kernels eagerly, and what it costs.",
        "Explain, in a few paragraphs, why a paged key-value cache beats a contiguous one for a "
        "serving engine, and what it costs.",
        "Explain, in a few paragraphs, why tensor parallelism needs an all-reduce at each block "
        "boundary, and what bounds the step.",
        "Explain, in a few paragraphs, why 4-bit weight quantization helps a memory-bound decode "
        "step, and where the accuracy goes.",
    ],
}

# Extra independent tasks for C6..C16. The original first four prompts and
# their rotation remain unchanged; wider phases must never repeat a prompt.
_EXTRA_TOPICS = {
    "prose": ["the Venetian Republic", "the development of printing", "the history of public libraries",
              "the rise of radio", "the development of railways", "the history of navigation",
              "the development of refrigeration", "the history of astronomy", "the development of bridges",
              "the history of postal systems", "the development of public parks", "the history of photography"],
    "code": ["a CSV reader with quoted fields", "a bounded thread-safe queue", "a trie with prefix search",
             "a streaming SHA-256 file verifier", "a topological sorter with cycle detection",
             "a retry loop with exponential backoff", "a command-line arithmetic parser",
             "a merge of sorted iterators", "a JSON Lines validator", "a sliding-window moving average",
             "a scheduler for periodic callbacks", "a filesystem path normalization function"],
    "json": ["books with title, author and publication_year", "planets and moons with name, parent and radius_km",
             "instruments with name, family and description", "languages with name, family and writing_system",
             "mountains with name, continent and height_m", "foods with name, origin and ingredients",
             "trees with name, habitat and leaf_type", "museums with name, city and specialty",
             "programming languages with name, year and paradigm", "algorithms with name, purpose and complexity",
             "sports with name, team_size and equipment", "space missions with name, year and destination"],
    "math": ["the sum of squares of the first n integers", "compound interest with monthly deposits",
             "the expected rolls until the first six", "a geometric series with ratio 2/3",
             "the derivative of x^x", "the volume of a cone from integration",
             "Bayes' rule for a test with 95 percent sensitivity", "the Euclidean algorithm for 1071 and 462",
             "the roots of x^3 - 6x^2 + 11x - 6", "counting paths on a 6 by 8 grid",
             "the variance of a binomial random variable", "the optimal dimensions of a fixed-volume cylinder"],
    "chat": ["prefix-cache eviction", "CPU cache coherence", "TCP congestion control", "floating-point rounding",
             "database write-ahead logs", "consistent hashing", "GPU shared memory", "memory mapping",
             "sparse matrix multiplication", "distributed consensus", "B-tree indexes", "backpressure in streams"],
}
_EXTRA_TEMPLATES = {
    "prose": "For {topic}, write a long detailed history, organized by era in full paragraphs.",
    "code": "Implement {topic} in a Python module, with type hints, docstrings and pytest tests.",
    "json": "List 25 different {topic} as a JSON array. Output only the JSON.",
    "math": "Explain and derive {topic} step by step, then work through two numerical examples and verify them.",
    "chat": "Explain {topic} in several detailed paragraphs, with examples, tradeoffs and practical applications.",
}
for _name, _topics in _EXTRA_TOPICS.items():
    CLASSES[_name].extend(_EXTRA_TEMPLATES[_name].format(topic=topic) for topic in _topics)
PROMPTS.extend(CLASSES["chat"][4:12])


def check_corpus():
    """Prompt 0 of each class must still be fabric_mtp_classes.sh's prompt."""
    import re as _re
    sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fabric_mtp_classes.sh")
    try:
        text = open(sh).read()
    except OSError:
        return  # run from outside the tree: nothing to check against
    fab = dict(_re.findall(r'^P\[(\w+)\]="(.*)"$', text, _re.M))
    for name, prompts in CLASSES.items():
        want = fab.get(name)
        if want is not None and prompts[0] != want:
            raise SystemExit(
                f"class {name!r} prompt 0 has drifted from fabric_mtp_classes.sh; "
                "the per-class numbers would not be comparable to the published tables")

def served_model(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("GET", "/v1/models")
    data = json.loads(conn.getresponse().read().decode())
    conn.close()
    ids = [m["id"] for m in data.get("data", [])]
    if not ids:
        raise SystemExit(f"no model served at {host}:{port}: {data}")
    return ids[0]


def stream_one(host, port, model, prompt, max_tokens, think, out, temperature=0, on_update=None):
    try:
        _stream_one(host, port, model, prompt, max_tokens, think, out, temperature, on_update)
    except Exception as error:
        out["error"] = str(error)


def _stream_one(host, port, model, prompt, max_tokens, think, out, temperature, on_update=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    conn = http.client.HTTPConnection(host, port, timeout=900)
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status == 400 and not think:
            # The template has no thinking knob: plain.
            conn.close()
            body.pop("chat_template_kwargs", None)
            conn = http.client.HTTPConnection(host, port, timeout=900)
            t0 = time.perf_counter()
            conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
        out.update(read_completion(resp, on_update=on_update))
    finally:
        conn.close()
    out["t0"] = t0
    out["end"] = time.perf_counter()
    out["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()


def phase(host, port, model, c, max_tokens, think, prompt_base, prompts=None, temperature=0, label="", reports=None):
    prompts = prompts or PROMPTS
    if c < 1 or c > len(prompts):
        raise ValueError(f"concurrency must be between 1 and {len(prompts)} distinct prompts")
    outs = [dict() for _ in range(c)]
    threads = []
    t_start = time.perf_counter()
    wall_start = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for i in range(c):
        prompt = prompts[(prompt_base + i) % len(prompts)]
        th = threading.Thread(target=stream_one,
                              args=(host, port, model, prompt, max_tokens, think, outs[i], temperature))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    t_end = time.perf_counter()
    wall_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"== {label}concurrency {c}: {wall_start} .. {wall_end}  wall {t_end - t_start:.1f} s")
    metrics = phase_metrics(outs, t_start, t_end)
    for i, o in enumerate(outs):
        n = o["tokens"]
        pace = ((o["last"] - o["first"]) / (o["chunks"] - 1) * 1e3) if o["chunks"] > 1 else float("nan")
        ttft = (o["first"] - o["t0"]) * 1e3 if o["first"] else float("nan")
        print(f"   req {i}: status {o['status']} finish={o['finish']} tokens {n} chunks {o['chunks']} "
              f"ttft {ttft:.0f} ms  client update pace {pace:.2f} ms/update")
    legacy = metrics["legacy_output_span_tokens_per_s"]
    print(f"   aggregate wall: {metrics['completion_tokens']} tokens in {metrics['wall_s']:.2f} s "
          f"= {metrics['wall_tokens_per_s']:.2f} tok/s (prefill included)")
    print(f"   legacy output-span aggregate: {legacy if legacy is not None else 'unavailable'} tok/s "
          "(initial TTFT excluded; later admissions included)")
    if reports is not None:
        reports.append({"class": label.strip() or "mixed", "concurrency": c,
                        "started_at": wall_start, "metrics": metrics, "requests": outs})
    return metrics["wall_tokens_per_s"]


def main(*, phase_runner=None, settle=None):
    # Counter-based clients can reconcile every request group, including warmup
    # and isolation, before starting the next measured phase.
    run_phase = phase if phase_runner is None else phase_runner
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("port", type=int)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--repeat", type=int, default=1, help="repeat each phase with identical prompts; report medians")
    ap.add_argument("--json-out", help="write raw requests and explicitly scoped rates for regression comparisons")
    ap.add_argument("--think", action="store_true", help="leave the template's thinking on")
    ap.add_argument("--warm", type=int, default=1, help="warm requests before the phases")
    ap.add_argument("--isolation", type=int, default=0, metavar="C",
                    help="the transcript-isolation check: prompt 0 alone, then beside C-1 others; its text must match")
    ap.add_argument("--classes", default=None, metavar="LIST",
                    help="run the sweep once per class (prose,code,json,math,chat), each phase drawing "
                         "its prompts from that class alone; 'all' runs every class")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 (default) is greedy; a positive value samples, for the pace at the card's defaults")
    args = ap.parse_args()
    cs = [int(x) for x in args.concurrency.split(",")]
    if (args.repeat < 1 or args.max_tokens < 1 or args.warm < 0 or not 0 <= args.isolation <= 16
            or any(c < 1 or c > 16 for c in cs)):
        ap.error("repeat/max-tokens must be positive; concurrency in [1, 16]; isolation in [0, 16]; warm >= 0")
    reports = []
    def save():
        if args.json_out:
            with open(args.json_out, "w") as output:
                json.dump({"schema_version": 1, "model": model, "max_tokens": args.max_tokens,
                           "temperature": args.temperature, "thinking": args.think,
                           "phases": reports}, output, indent=2, allow_nan=False)
    if args.classes:
        check_corpus()
    model = served_model(args.host, args.port)
    print(f"model {model}")
    if settle is not None:
        settle(args.host, args.port, [])
    for w in range(args.warm):
        o = {}
        stream_one(args.host, args.port, model, PROMPTS[w % len(PROMPTS)], 64, args.think, o)
        if o.get("error"):
            raise RuntimeError(o["error"])
        if settle is not None:
            settle(args.host, args.port, [o])
        print(f"warm {w}: status {o['status']} tokens {o['tokens']}")
    if args.isolation > 1:
        alone = {}
        stream_one(args.host, args.port, model, PROMPTS[0], args.max_tokens, args.think, alone)
        if alone.get("error"):
            raise RuntimeError(alone["error"])
        if settle is not None:
            settle(args.host, args.port, [alone])
        outs = [dict() for _ in range(args.isolation)]
        threads = [threading.Thread(target=stream_one,
                                    args=(args.host, args.port, model, PROMPTS[i % len(PROMPTS)], args.max_tokens,
                                          args.think, outs[i]))
                   for i in range(args.isolation)]
        for th in threads: th.start()
        for th in threads: th.join()
        for out in outs:
            if out.get("error"):
                raise RuntimeError(out["error"])
        if settle is not None:
            settle(args.host, args.port, outs)
        same = alone["text"] == outs[0]["text"] and alone["tokens"] == outs[0]["tokens"]
        print(f"== isolation: prompt 0 alone ({alone['tokens']} tokens) vs beside {args.isolation - 1} others "
              f"({outs[0]['tokens']} tokens): {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            a, b = alone["text"], outs[0]["text"]
            k = 0
            while k < len(a) and k < len(b) and a[k] == b[k]: k += 1
            print(f"   first difference at char {k}: alone {a[k:k+60]!r} | batched {b[k:k+60]!r}")
            raise SystemExit("transcript isolation failed")
    draw = "greedy" if args.temperature == 0 else f"sampled T={args.temperature}"
    if args.classes:
        names = list(CLASSES) if args.classes == "all" else args.classes.split(",")
        for n in names:
            if n not in CLASSES:
                raise SystemExit(f"unknown class {n!r}; known: {', '.join(CLASSES)}")
        table = {}
        for n in names:
            base = 0
            for c in cs:
                rates = [run_phase(args.host, args.port, model, c, args.max_tokens, args.think, base,
                               CLASSES[n], args.temperature, f"{n} ", reports) for _ in range(args.repeat)]
                table[(n, c)] = statistics.median(rates)
                base += c
                save()
        print(f"== per-class wall aggregate tokens/s (client wall time: prefill and queueing included — the engine's decode-only rate is timed_load's table), {draw}, max_tokens {args.max_tokens}")
        print("| class | " + " | ".join(f"c={c}" for c in cs) + " |")
        print("|---|" + "---|" * len(cs))
        for n in names:
            print(f"| {n} | " + " | ".join(f"{table[(n, c)]:.1f}" for c in cs) + " |")
        return
    base = 0
    summary = []
    for c in cs:
        rate = statistics.median(run_phase(args.host, args.port, model, c, args.max_tokens, args.think, base,
                                       None, args.temperature, reports=reports) for _ in range(args.repeat))
        summary.append((c, rate))
        base += c
        save()
    print(f"== summary (wall, {draw}): " + "  ".join(f"c={c}: {r:.1f} tok/s" for c, r in summary))


if __name__ == "__main__":
    main()
