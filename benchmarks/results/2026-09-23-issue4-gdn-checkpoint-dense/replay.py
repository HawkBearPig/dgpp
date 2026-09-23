#!/usr/bin/env python3
"""Replay the unchanged issue #4 request on an already reserved, fresh server."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def score(response, oracle):
    choice = response["choices"][0]
    actual = json.loads(choice["message"]["content"], object_pairs_hook=unique_object)
    if not isinstance(actual, dict):
        raise ValueError("assistant content is not a JSON object")
    checks = {key: key in actual and actual[key] == value for key, value in oracle.items()}
    usage = response.get("usage", {})
    return {
        "checks": checks,
        "correct": sum(checks.values()),
        "total": len(checks),
        "actual": actual,
        "expected": oracle,
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
        "semantic_pass": all(checks.values()) and set(actual) == set(oracle),
        "protocol_pass": choice.get("finish_reason") == "stop"
        and usage.get("prompt_tokens") == 261120
        and (usage.get("prompt_tokens_details") or {}).get("cached_tokens") == 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18084")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    payload = args.request.read_bytes()
    request = json.loads(payload)
    oracle = json.loads(args.oracle.read_bytes())
    (args.out / "request.json").write_bytes(payload)
    (args.out / "oracle.json").write_bytes(args.oracle.read_bytes())

    def get(path, filename):
        with urllib.request.urlopen(args.url.rstrip("/") + path, timeout=10) as reply:
            body = reply.read()
        (args.out / filename).write_bytes(body)
        return json.loads(body)

    models = get("/v1/models", "models.json")
    if request["model"] not in {x["id"] for x in models["data"]}:
        raise RuntimeError("server does not advertise the requested checkpoint")
    before = get("/metrics", "metrics-before.json")
    if any(before["scheduler"][k] != 0 for k in ("active", "queued", "prompts_prefilled")):
        raise RuntimeError("a fresh, idle server is required")
    if before["service"]["requests_total"] != 0:
        raise RuntimeError("a prior generation request was admitted")
    start = time.monotonic()
    transport = {"request_sha256": hashlib.sha256(payload).hexdigest(),
                 "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as reply:
            body = reply.read()
            transport["status"] = reply.status
    except urllib.error.HTTPError as error:
        body = error.read()
        transport["status"] = error.code
    finally:
        transport["wall_seconds"] = time.monotonic() - start
        (args.out / "transport.json").write_text(json.dumps(transport, indent=2) + "\n")
    (args.out / "response.json").write_bytes(body)
    after = get("/metrics", "metrics-after.json")
    result = score(json.loads(body), oracle)
    result["first_request_verified"] = after["service"]["requests_total"] == 1 and after["scheduler"]["prompts_prefilled"] == 1
    result["pass"] = transport["status"] == 200 and result["semantic_pass"] and result["protocol_pass"] and result["first_request_verified"]
    (args.out / "verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
