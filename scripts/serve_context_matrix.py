#!/usr/bin/env python3
"""Measure C1 decode at fixed context buckets on an otherwise idle server.

Use the tools venv (tokenizers is required). Each bucket uses a distinct,
deterministic parcel document: one cold request, then two identical requests.
The 0 bucket is a minimal prompt, not a literal zero-token engine state.
"""
import argparse
import hashlib
import json
from pathlib import Path

from cluster_doctor import cache_root, cached_snapshot
from qwen_yarn_release_check import Lane, filler, save_record
from site_env import cache_environment
from timed_load import COUNTERS, reconciled_metrics


def make_prompt(tokenizer, target, tag, seed):
    if target == 0:
        # Keep the minimal context below 256 tokens even with MiniMax's
        # approximately 190-token chat wrapper. A bounded tag keeps long
        # deployment labels from turning this into a larger context bucket.
        identity = hashlib.sha256(f'{tag}:{seed}'.encode()).hexdigest()[:12]
        return f'Context {identity}. Write numbers 1 through 200, one per line.'
    prefix = f"Context measurement {tag}, bucket {target}.\n"
    suffix = ("\nWrite a numbered list describing the first 200 parcels and their cities. "
              "Continue until you have listed all 200.")
    # Reserve space for the chat wrapper, the 256 output tokens and draft
    # verification. This also lets a 256K pool run the 256K bucket safely.
    budget = target - 512
    lines = filler(seed, target // 20 + 1)
    low, high = 0, len(lines)
    while low < high:
        mid = (low + high + 1) // 2
        prompt = prefix + "".join(lines[:mid]) + suffix
        if len(tokenizer.encode(prompt, add_special_tokens=False).ids) <= budget:
            low = mid
        else:
            high = mid - 1
    return prefix + "".join(lines[:low]) + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("--model", required=True, help="checkpoint ID, not a serving alias")
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[0, 32768, 65536, 131072, 262144, 524288])
    parser.add_argument("--tag", required=True, help="unique per server launch")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    snapshot = cached_snapshot(args.model, cache_root(cache_environment()))
    tokenizer_path = snapshot / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    lane = Lane(f"http://{args.host}:{args.port}")
    limit = lane.reported_limit()
    if not isinstance(limit, int) or limit <= 0:
        raise RuntimeError("server must advertise its request context limit")
    report = {"schema_version": 1, "model": args.model, "surface": lane.surface,
              "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
              "seed": 20260922, "tag": args.tag, "max_tokens": 256,
              "thinking": "server_default", "samples": [], "skipped": []}
    before = reconciled_metrics(lane.host, lane.port)
    for target in args.lengths:
        if target > limit:
            report["skipped"].append({"target": target, "reason": "exceeds request limit",
                                      "limit": limit})
            save_record(args.json_out, report)
            continue
        prompt = make_prompt(tokenizer, target, args.tag, report["seed"])
        for repeat in range(3):
            answer = lane.ask(prompt, 256, timeout=14400)
            after = reconciled_metrics(lane.host, lane.port, before, [answer])
            delta = {name: after[name] - before[name] for name in COUNTERS}
            usage = answer["usage"]
            actual = usage["prompt_tokens"]
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if target and not target - 1024 <= actual <= target - 256:
                raise RuntimeError(f"bucket {target}: unexpected actual prompt length {actual}")
            if not target and actual > 256:
                raise RuntimeError(f"minimal prompt is too long: {actual}")
            if repeat == 0 and cached:
                raise RuntimeError("cold context request reused cached tokens")
            if (delta["prompts_prefilled"] != 1 or
                    delta["prompt_tokens_computed"] != actual - cached or
                    delta["decode_rows"] != delta["decode_steps"] or
                    delta["decode_steps"] <= 0 or delta["step_ms"] <= 0):
                raise RuntimeError("counters do not describe one completed C1 decode")
            generated = delta["tokens_generated"] - 1
            sample = {"requested_tokens": target, "repeat": repeat, "engine": delta,
                      "engine_tokens_per_s": generated * 1000 / delta["step_ms"],
                      "ms_per_pass": delta["step_ms"] / delta["decode_steps"],
                      "tokens_per_pass": generated / delta["decode_steps"], "answer": answer}
            report["samples"].append(sample)
            save_record(args.json_out, report)
            print(f"{target} r{repeat}: {actual} prompt tokens, {cached} cached, "
                  f"prefill {delta['prefill_ms'] / 1000:.3f}s, "
                  f"decode {sample['engine_tokens_per_s']:.2f} tok/s, "
                  f"{sample['ms_per_pass']:.2f} ms/pass", flush=True)
            before = after


if __name__ == "__main__":
    main()
