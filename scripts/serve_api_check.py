#!/usr/bin/env python3
"""The request-field check against a running service (2026-09-06): stop,
n, logit_bias and the usage details, exercised on the real model and
reported compactly.

    serve_api_check.py HOST PORT [MODEL] [--tokenizer PATH]

  stop        a one-shot with a stop string the model is bound to produce
              (the second word of its own unconstrained answer): the content
              must end before it, finish_reason "stop"; then the same
              streamed, and the deltas must never show the stop string.
  n           n=2 at temperature 0.8 with a fixed seed: two indexed choices,
              the usage's completion_tokens the sum; streamed, one [DONE]
              and one usage chunk.
  logit_bias  a token forced (+100): every generated token must be that
              token; a ban on a token the answer never used changes
              nothing; a malformed table is refused by name.
  usage       prompt_tokens_details.cached_tokens on a repeated prompt
              (the prefix cache's attach) and completion_tokens_details.
              reasoning_tokens on a thinking answer.
Exits nonzero when any check fails.
"""
import json
from serve_client import served_model
import sys
import http.client
import uuid
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("host")
parser.add_argument("port", type=int)
parser.add_argument("model", nargs="?")
parser.add_argument("--tokenizer", type=Path, help="checkpoint tokenizer.json; selects a visible token for logit_bias")
args = parser.parse_args()
HOST, PORT = args.host, args.port
MODEL = args.model or served_model(HOST, PORT)
FORCE_ID, FORCE_TEXT = 100, None
if args.tokenizer:
    tokenizer = json.loads(args.tokenizer.read_text())
    special = {t["id"] for t in tokenizer.get("added_tokens", []) if t.get("special")}
    # These ASCII byte-level BPE entries decode literally. Low ids can be
    # invisible control tokens (Dory reserves 0..999), so inspect the vocab.
    for text in ("!", "~", "^"):
        token = tokenizer["model"]["vocab"].get(text)
        if token is not None and token not in special:
            FORCE_ID, FORCE_TEXT = token, text
            break
    else:
        parser.error("tokenizer has no regular ASCII token for the logit_bias probe")
failures = []


def post(path, body, stream=False):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if not stream:
        raw = resp.read().decode("utf-8", "replace")
        conn.close()
        try:
            return resp.status, json.loads(raw)
        except json.JSONDecodeError:
            return resp.status, raw
    events = []
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if line.startswith(b"data: "):
                    payload = line[6:].decode("utf-8", "replace")
                    events.append(payload if payload == "[DONE]" else json.loads(payload))
    conn.close()
    return resp.status, events


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


def chat(messages, **fields):
    # This template always opens a think block; a low reasoning effort keeps
    # it short and the checks read the content past it.
    body = {"model": MODEL, "messages": messages, "max_tokens": 160, "temperature": 0,
            "reasoning_effort": "low"}
    body.update(fields)
    return body


def content_of(resp):
    return resp["choices"][0]["message"].get("content") or ""


prompt = [{"role": "user", "content": "Name three primary colors, one per line, nothing else."}]

# --- the unconstrained answer: the material for stop and the ban -----------
# Thinking off for the stop material (the templates that read
# enable_thinking — Qwen3.8-Flash-Next, GLM-4.7 — answer directly; GLM-5.3-
# Flash's ignores it and keeps its short low-effort block): a model that
# spends the 160-token budget inside its think block leaves no words to
# build a stop sequence from (GLM-4.7, 2026-09-10).
# Explicit thinking controls must agree with reasoning_effort. Otherwise the
# fallback below mistakes a conflicting request for an unsupported template knob.
no_think = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
st, base = post("/v1/chat/completions", chat(prompt, **no_think))
if st == 400:  # the template has no such knob (GLM-5.3-Flash): plain
    no_think = {}
    st, base = post("/v1/chat/completions", chat(prompt))
if st != 200:
    print("baseline request failed:", st, base)
    sys.exit(2)
base_text = content_of(base)
words = base_text.split()
print(f"baseline: {base_text!r} ({base['usage']})")

# --- stop -------------------------------------------------------------------
if len(words) >= 2:
    stop = words[1]
    st, r = post("/v1/chat/completions", chat(prompt, stop=stop, **no_think))
    text = content_of(r)
    check("stop one-shot",
          st == 200 and stop not in text and text == base_text[: base_text.index(stop)]
          and r["choices"][0]["finish_reason"] == "stop"
          and r["usage"]["completion_tokens"] < base["usage"]["completion_tokens"],
          f"stop={stop!r} content={text!r} finish={r['choices'][0]['finish_reason']} usage={r['usage']}")
    st, ev = post("/v1/chat/completions", chat(prompt, stop=[stop], stream=True,
                                               stream_options={"include_usage": True}, **no_think), stream=True)
    deltas = [e["choices"][0]["delta"].get("content", "") for e in ev
              if e != "[DONE]" and e.get("choices")]
    joined = "".join(d for d in deltas if d)
    finishes = [e["choices"][0]["finish_reason"] for e in ev if e != "[DONE]" and e.get("choices")]
    usage = [e["usage"] for e in ev if e != "[DONE]" and e.get("usage")]
    check("stop stream",
          st == 200 and joined == text and all(stop not in (d or "") for d in deltas)
          and finishes[-1] == "stop" and ev[-1] == "[DONE]" and len(usage) == 1,
          f"joined={joined!r} finish={finishes[-1] if finishes else None} usage={usage}")
else:
    check("stop", False, "the baseline answer has fewer than two words")

# --- n ----------------------------------------------------------------------
st, r = post("/v1/chat/completions", chat(prompt, n=2, temperature=0.8, seed=7, max_tokens=96))
choices = r.get("choices", []) if st == 200 else []
texts = [(c["message"].get("content") or c["message"].get("reasoning_content") or "")
         for c in choices]
check("n=2 one-shot",
      st == 200 and [c["index"] for c in choices] == [0, 1]
      and r["usage"]["completion_tokens"] <= 192 and all(t for t in texts),
      f"indices={[c.get('index') for c in choices]} usage={r.get('usage')} texts={[t[:40] for t in texts]}")
st, ev = post("/v1/chat/completions", chat(prompt, n=2, temperature=0.8, seed=7, max_tokens=96,
                                           stream=True, stream_options={"include_usage": True}), stream=True)
by_index = {0: "", 1: ""}
finishes = {}
for e in ev:
    if e == "[DONE]" or not e.get("choices"):
        continue
    c = e["choices"][0]
    by_index[c["index"]] = by_index.get(c["index"], "") + (
        c["delta"].get("content") or c["delta"].get("reasoning_content") or "")
    if c.get("finish_reason"):
        finishes[c["index"]] = c["finish_reason"]
usage = [e["usage"] for e in ev if e != "[DONE]" and e.get("usage")]
check("n=2 stream",
      st == 200 and by_index[0] and by_index[1] and set(finishes) == {0, 1}
      and ev.count("[DONE]") == 1 and len(usage) == 1,
      f"choice0={by_index[0][:40]!r} choice1={by_index[1][:40]!r} finishes={finishes} usage={usage}")

# --- logit_bias -------------------------------------------------------------
# The service does not expose token ids (the logprobs carry text and
# bytes), so the check forces one id with +100 — every generated token must
# be that one — bans an id the answer never used (nothing changes) and
# sends a malformed table (refused by name). Use plain completions for the
# forced-token probe: forcing a token inside an open chat reasoning block
# never closes that block, and Chat logprobs cover visible content only.
# The unit and loopback gates cover the ban's exact arithmetic.
forced = {"model": MODEL, "prompt": "Continue: ", "temperature": 0,
          "logit_bias": {str(FORCE_ID): 100}, "max_tokens": 6}
st, r = post("/v1/completions", forced)
forced_text = r["choices"][0]["text"] if st == 200 else str(r)
st2, r2 = post("/v1/completions", dict(forced, logprobs=1))
toks = r2["choices"][0]["logprobs"]["tokens"] if st2 == 200 else []
check("logit_bias force",
      st == 200 and st2 == 200 and len(set(toks)) == 1 and len(toks) == 6
      and (FORCE_TEXT is None or forced_text == FORCE_TEXT * 6)
      and forced_text == r2["choices"][0]["text"],
      f"tokens={toks} text={forced_text!r}")
st, r = post("/v1/chat/completions", chat(prompt, logit_bias={str(FORCE_ID): -100}, **no_think))
check("logit_bias ban",
      st == 200 and content_of(r) == base_text,
      f"a ban on an unused token changes nothing: {content_of(r)[:40]!r}")
st, r = post("/v1/chat/completions", chat(prompt, logit_bias={"x": 1}))
check("logit_bias refused by name", st == 400 and r.get("error", {}).get("param") == "logit_bias",
      f"status={st} error={r.get('error') if isinstance(r, dict) else r}")

# --- usage details ----------------------------------------------------------
# A fresh leading marker keeps this cold/hot comparison valid when the
# smoke test is repeated against the same running server. The prompt runs
# past the cache's entry floor (engine.prefix_min_tokens, 1024 by default):
# a shorter prompt attaches to nothing and takes no snapshot slot.
long_prompt = [{"role": "user", "content": str(uuid.uuid4()) + "\n" + ("Background note. " * 700)
                + "\nSummarize the causes of the fall of the Roman Republic in four sentences."}]
st, a = post("/v1/chat/completions", chat(long_prompt, max_tokens=96))
st2, b = post("/v1/chat/completions", chat(long_prompt, max_tokens=96))
ua, ub = a.get("usage", {}), b.get("usage", {})
check("usage cached_tokens",
      st == 200 and st2 == 200
      and ua.get("prompt_tokens_details", {}).get("cached_tokens") == 0
      and ub.get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0,
      f"cold={ua.get('prompt_tokens_details')} hot={ub.get('prompt_tokens_details')}")
check("usage reasoning_tokens",
      ua.get("completion_tokens_details", {}).get("reasoning_tokens", 0) > 0
      and ua["completion_tokens_details"]["reasoning_tokens"] <= ua["completion_tokens"],
      f"{ua.get('completion_tokens_details')} of {ua.get('completion_tokens')} completion tokens")

print("FAILED:" if failures else "ALL OK", failures)
sys.exit(1 if failures else 0)
