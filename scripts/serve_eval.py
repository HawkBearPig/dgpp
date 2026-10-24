#!/usr/bin/env python3
"""Task-level eval of a running dgpp-serve world (2026-09-08): the pass rates
that perplexity cannot give — code that runs, arithmetic that is right,
structured extraction that is exact — for the checkpoint the world serves,
so deployments can be compared on the same items with the same settings.

    serve_eval.py HOST PORT --out DIR [--tasks humaneval,gsm8k,extract]
        [--limit N] [--concurrency C] [--max-tokens N] [--retry-max-tokens N]
        [--reasoning-effort low|medium|high] [--data DIR] [--seed N]

Tasks (greedy, temperature 0, the served model from GET /v1/models):
  humaneval  164 problems (DGPP_DATA_DIR/HumanEval.jsonl, the original
             openai/human-eval release): the model completes the function;
             the fenced code (or the whole reply) runs against the problem's
             tests in a subprocess with a 10 s timeout; pass = exit 0.
  gsm8k      the test split (DGPP_DATA_DIR/gsm8k_test.jsonl, first
             --limit items, default 300): step-by-step then "#### <number>";
             pass = the last number in the reply equals the gold answer.
  extract    synthetic records (name, age, city, email, order total)
             rendered as a sentence, extracted through response_format
             json_schema (the server's constrained decoding guarantees the
             shape; the pass is the CONTENT: every field exact) — 100 items
             from --seed, identical across configs.
Writes <out>/<task>.jsonl (one line per item: id, pass, reply, tokens),
<out>/summary.json, and prints the summary. Every item is independent, so
--concurrency C keeps C requests in flight.

The token cap and the retry (2026-10-22): every item is first asked under
the standard output cap (--max-tokens; 512 for extraction), and the pass
rate is that budget's verdict — the comparable number across checkpoints.
An item the server cut at the cap (finish_reason "length") is then asked
once more at --retry-max-tokens (0 disables the retry); its record keeps
the retry under "retry" (reply, tokens, verdict) and the summary counts,
per task, the retried items, those that completed within the larger cap
and those that then passed. A model that reaches the right answer only
with more tokens is a behaviour worth seeing, not a hidden failure — and
the standard-budget score is never changed by it.
"""
import argparse
import concurrent.futures
import http.client
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from data_paths import data_dir, require_file
from humaneval_program import humaneval_program

ap = argparse.ArgumentParser()
ap.add_argument("host")
ap.add_argument("port", type=int)
ap.add_argument("--out", required=True)
ap.add_argument("--tasks", default="humaneval,gsm8k,extract")
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--concurrency", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--retry-max-tokens", type=int, default=8192,
                help="the larger cap a response cut at the standard cap is asked once more under (0: no retry)")
ap.add_argument("--reasoning-effort", default="low")
ap.add_argument("--no-think", action="store_true",
                help="chat_template_kwargs.enable_thinking=false (the templates that read it: "
                     "Qwen3.8-Flash-Next, GLM-4.7 — GLM-4.7 ignores reasoning_effort and thinks to the cap otherwise)")
ap.add_argument("--data", default=None)
ap.add_argument("--allow-code-execution", action="store_true", help="acknowledge that HumanEval runs generated Python without a security sandbox")
ap.add_argument("--seed", type=int, default=20260908)
ap.add_argument("--model", default=None, help="defaults to the served id")
args = ap.parse_args()
args.data = args.data or str(data_dir())
tasks = args.tasks.split(",")
if "humaneval" in tasks and not args.allow_code_execution:
    ap.error("HumanEval executes generated code locally. Use an isolated environment and explicitly pass --allow-code-execution, or select --tasks gsm8k,extract.")
for task, name in (("humaneval", "HumanEval.jsonl"), ("gsm8k", "gsm8k_test.jsonl")):
    if task in tasks:
        require_file(os.path.join(args.data, name))
os.makedirs(args.out, exist_ok=True)


def request(path, body=None, timeout=900):
    conn = http.client.HTTPConnection(args.host, args.port, timeout=timeout)
    if body is None:
        conn.request("GET", path)
    else:
        conn.request("POST", path, body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"{path}: {resp.status} {data[:300]}")
    return json.loads(data)


MODEL = args.model or request("/v1/models")["data"][0]["id"]


def chat(messages, max_tokens=None, response_format=None):
    body = {
        "model": MODEL, "messages": messages, "temperature": 0,
        "max_tokens": max_tokens or args.max_tokens,
        # The service refuses an effort beside the explicit thinking switch
        # (the template ignores it when thinking is off): send one or the other.
        **({"reasoning_effort": args.reasoning_effort} if not args.no_think else {}),
        **({"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else {}),
    }
    if response_format is not None:
        body["response_format"] = response_format
    t0 = time.perf_counter()
    r = request("/v1/chat/completions", body)
    choice = r["choices"][0]
    usage = r.get("usage", {})
    return {
        "content": choice["message"].get("content") or "",
        "finish": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def fenced_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else text


# ---- humaneval --------------------------------------------------------------
def humaneval_items():
    path = os.path.join(args.data, "HumanEval.jsonl")
    return [json.loads(l) for l in open(path)][: args.limit]


def humaneval_run(item, max_tokens=None):
    prompt = ("Complete the following Python function. Reply with the complete "
              "function (imports, signature and body) in one ```python code block "
              "and nothing else.\n\n```python\n" + item["prompt"] + "```")
    r = chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
    code = fenced_code(r["content"])
    program = humaneval_program(item, code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, timeout=10)
        ok = proc.returncode == 0
        err = proc.stderr.decode()[-300:]
    except subprocess.TimeoutExpired:
        ok, err = False, "timeout"
    finally:
        os.unlink(path)
    return dict(id=item["task_id"], ok=ok, err=err if not ok else "", **r)


# ---- gsm8k ------------------------------------------------------------------
def gsm8k_items():
    path = os.path.join(args.data, "gsm8k_test.jsonl")
    return [json.loads(l) for l in open(path)][: args.limit]


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text):
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        m = NUM.findall(text)
    if not m:
        return None
    s = m[-1].replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_run(i_item, max_tokens=None):
    i, item = i_item
    gold = float(item["answer"].split("####")[-1].strip().replace(",", ""))
    prompt = (item["question"] + "\n\nSolve this step by step, then give the "
              "final numeric answer on its own line as: #### <number>")
    r = chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
    got = last_number(r["content"])
    ok = got is not None and abs(got - gold) < 1e-6
    return dict(id=f"gsm8k-{i}", ok=ok, gold=gold, got=got, **r)


# ---- extract ----------------------------------------------------------------
FIRST = ["Ada", "Ben", "Chloe", "Dmitri", "Esme", "Farid", "Greta", "Hiro", "Ines", "Jonas",
         "Kavya", "Leon", "Maren", "Nikolai", "Oona", "Priya", "Quentin", "Rosa", "Sven", "Tamsin"]
LAST = ["Okafor", "Lindqvist", "Marchetti", "Nakamura", "Petrov", "Quiroga", "Rasmussen", "Sato",
        "Tremblay", "Ullmann", "Varga", "Whitcombe", "Xiang", "Yilmaz", "Zapata", "Abernathy"]
CITY = ["Lisbon", "Tallinn", "Kyoto", "Montevideo", "Tromsø", "Valletta", "Hobart", "Zagreb",
        "Cusco", "Galway", "Nairobi", "Ljubljana"]
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 130},
        "city": {"type": "string"},
        "email": {"type": "string"},
        "order_total": {"type": "number"},
    },
    "required": ["name", "age", "city", "email", "order_total"],
    "additionalProperties": False,
}


def extract_items():
    rng = random.Random(args.seed)
    items = []
    for i in range(min(args.limit, 100)):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        age = rng.randint(19, 87)
        city = rng.choice(CITY)
        email = f"{name.split()[0].lower()}.{name.split()[1].lower()}{rng.randint(1, 99)}@example.org"
        total = round(rng.uniform(5, 950), 2)
        forms = [
            f"Order #{rng.randint(1000, 9999)}: {name} ({age} years old, living in {city}) paid {total:.2f} and asked for the receipt at {email}.",
            f"Customer record — name: {name}; contact: {email}; age {age}; city of residence {city}; last order total {total:.2f}.",
            f"{name}, who is {age} and based in {city}, can be reached at {email}. Their most recent purchase came to {total:.2f}.",
        ]
        items.append(dict(id=f"extract-{i}", text=rng.choice(forms),
                          gold=dict(name=name, age=age, city=city, email=email, order_total=total)))
    return items


EXTRACT_MAX_TOKENS = 512  # the standard cap of the extraction items


def extract_run(item, max_tokens=None):
    prompt = ("Extract the customer's name, age, city, email and order_total from the text "
              "below as JSON.\n\nText: " + item["text"])
    rf = {"type": "json_schema", "json_schema": {"name": "customer", "schema": SCHEMA, "strict": True}}
    r = chat([{"role": "user", "content": prompt}], max_tokens=max_tokens or EXTRACT_MAX_TOKENS, response_format=rf)
    try:
        got = json.loads(r["content"])
    except json.JSONDecodeError:
        got = None
    g = item["gold"]
    ok = (got is not None and got.get("name") == g["name"] and got.get("age") == g["age"]
          and got.get("city") == g["city"] and got.get("email") == g["email"]
          and abs(float(got.get("order_total", -1)) - g["order_total"]) < 0.005)
    return dict(id=item["id"], ok=ok, gold=g, got=got, **r)


TASKS = {
    "humaneval": (humaneval_items, humaneval_run),
    "gsm8k": (lambda: list(enumerate(gsm8k_items())), gsm8k_run),
    "extract": (extract_items, extract_run),
}

summary = {"model": MODEL, "settings": dict(max_tokens=args.max_tokens, extract_max_tokens=EXTRACT_MAX_TOKENS,
                                            retry_max_tokens=args.retry_max_tokens,
                                            reasoning_effort=args.reasoning_effort, no_think=args.no_think,
                                            limit=args.limit, seed=args.seed), "tasks": {}}
for task in args.tasks.split(","):
    items_fn, run_fn = TASKS[task]
    items = items_fn()
    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for res in pool.map(run_fn, items):
            results.append(res)
            done = len(results)
            if done % 20 == 0 or done == len(items):
                passed = sum(1 for x in results if x["ok"])
                print(f"  {task}: {done}/{len(items)} done, {passed} pass, {time.time() - t0:.0f}s", flush=True)
    # The retry: every response the server cut at the standard cap is asked
    # once more at the larger cap. The verdict under the standard cap stands
    # in `ok`; the retry's record (reply, tokens, its own verdict) goes under
    # "retry" and the summary counts how many completed and passed there.
    capped = [i for i, x in enumerate(results) if x.get("finish") == "length"]
    retry_cap = args.retry_max_tokens
    if capped and retry_cap > 0:
        t1 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            retries = list(pool.map(lambda i: run_fn(items[i], max_tokens=retry_cap), capped))
        for i, r in zip(capped, retries):
            results[i]["retry"] = dict(max_tokens=retry_cap, **{k: v for k, v in r.items() if k not in ("id", "gold")})
        completed = sum(1 for r in retries if r.get("finish") == "stop")
        print(f"  {task}: {len(capped)} at the {('%d' % EXTRACT_MAX_TOKENS) if task == 'extract' else ('%d' % args.max_tokens)}-token cap retried at "
              f"{retry_cap}: {completed} completed, {sum(1 for r in retries if r['ok'])} pass, {time.time() - t1:.0f}s", flush=True)
    with open(os.path.join(args.out, f"{task}.jsonl"), "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    n = len(results)
    passed = sum(1 for x in results if x["ok"])
    toks = [x["completion_tokens"] for x in results if x.get("completion_tokens")]
    retried = [x["retry"] for x in results if x.get("retry")]
    summary["tasks"][task] = dict(n=n, passed=passed, rate=round(passed / n, 4) if n else None,
                                  seconds=round(time.time() - t0),
                                  mean_completion_tokens=round(sum(toks) / len(toks)) if toks else None,
                                  truncated=len(capped),
                                  retry_max_tokens=retry_cap if capped and retry_cap > 0 else None,
                                  retried=len(retried),
                                  retry_completed=sum(1 for r in retried if r.get("finish") == "stop"),
                                  retry_passed=sum(1 for r in retried if r.get("ok")))
    st = summary["tasks"][task]
    print(f"== {task}: {passed}/{n} = {100 * passed / n:.1f} %  ({st['seconds']} s, "
          f"mean {st['mean_completion_tokens']} completion tokens, {st['truncated']} at the cap"
          + (f"; retried at {st['retry_max_tokens']}: {st['retry_completed']} completed, {st['retry_passed']} pass" if st['retried'] else "")
          + ")", flush=True)
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
print(json.dumps(summary["tasks"]))
