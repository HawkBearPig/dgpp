#!/usr/bin/env python3
"""Replay interrupted chat histories against a running, otherwise idle server.

Keeps exactly the content/reasoning delivered over SSE, sends it back through
chat rendering/tokenization, and compares cached and explicitly cold replies.
The complete JSON report includes request ids, histories, usage, TTFT, engine
prefill work and cache/retirement counters. This is a correctness diagnostic,
not a throughput benchmark. Start/stop and rank/swap checks belong to its caller.
"""
import argparse
import copy
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from serve_client import default_url, model_at_url


class Probe:
    def __init__(self, url, timeout=600, thinking=False):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("expected an http:// endpoint")
        self.host, self.port, self.timeout = parsed.hostname, parsed.port or 80, timeout
        self.thinking = thinking
        self.model = model_at_url(url)

    def metrics(self):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            connection.request("GET", "/v1/metrics")
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"metrics returned HTTP {response.status}")
            return json.loads(response.read())
        finally:
            connection.close()

    def drained(self, before=None, cancelled=False):
        deadline = time.monotonic() + self.timeout
        while True:
            metrics = self.metrics()
            sched, service = metrics["scheduler"], metrics["service"]
            if service.get("engine_failed"):
                raise RuntimeError("the engine failed")
            retired = before is None or (sched["prompts_prefilled"] > before["scheduler"]["prompts_prefilled"] and
                service["requests_cancelled"] == before["service"]["requests_cancelled"] + int(cancelled))
            if not sched["active"] and not sched["queued"] and not sched.get("prefilling", 0) and retired:
                if before and service["requests_total"] != before["service"]["requests_total"] + 1:
                    raise RuntimeError("metrics include another request; run on an idle world")
                return metrics
            if time.monotonic() >= deadline:
                raise RuntimeError("request did not drain with the expected retirement reason")
            time.sleep(.02)

    def chat(self, messages, max_tokens, *, cache=True, cancel_deltas=0, ignore_eos=False):
        before = self.drained()
        template_kwargs = {"clear_thinking": False}
        if not self.thinking:
            template_kwargs["enable_thinking"] = False
        payload = {"model": self.model, "messages": messages, "temperature": 0, "seed": 157,
                   "max_tokens": max_tokens, "stream": True, "prefix_cache": cache,
                   "ignore_eos": ignore_eos, "stream_options": {"include_usage": True},
                   "chat_template_kwargs": template_kwargs}
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        response = None
        started = time.monotonic()
        result = {"id": None, "content": "", "reasoning_content": "", "usage": None,
                  "finish": None, "ttft_ms": None, "deltas": 0, "cancelled": False}
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"chat returned HTTP {response.status}: {response.read()[:2000]!r}")
            while True:
                line = response.readline()
                if not line:
                    raise RuntimeError("SSE ended without [DONE]")
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    break
                event = json.loads(line[6:])
                if "error" in event:
                    raise RuntimeError(f"stream error: {event['error']}")
                result["id"] = event.get("id", result["id"])
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    visible = bool(delta.get("content") or delta.get("reasoning_content"))
                    for field in ("content", "reasoning_content"):
                        result[field] += delta.get(field) or ""
                    if visible:
                        result["deltas"] += 1
                        if result["ttft_ms"] is None:
                            result["ttft_ms"] = 1000 * (time.monotonic() - started)
                    if choice.get("finish_reason"):
                        result["finish"] = choice["finish_reason"]
                if cancel_deltas and result["deltas"] >= cancel_deltas:
                    if result["finish"]:
                        raise RuntimeError("answer finished before the planned disconnect")
                    result["cancelled"] = True
                    break
        finally:
            if response is not None:
                response.close()
            connection.close()
        if cancel_deltas and not result["cancelled"]:
            raise RuntimeError("answer finished before enough deltas arrived")
        if not cancel_deltas and result["usage"] is None:
            raise RuntimeError("completed answer did not include usage")
        after = self.drained(before, result["cancelled"])
        result["before"], result["after"] = before, after
        fields = ("prompt_tokens", "prompt_tokens_computed", "tokens_generated", "prefill_ms", "decode_steps")
        result["engine"] = {key: after["scheduler"][key] - before["scheduler"][key] for key in fields}
        if after["service"]["requests_shed_pool"] != before["service"]["requests_shed_pool"]:
            raise RuntimeError("the probe shed a request under pool pressure")
        if result["usage"]:
            usage = result["usage"]
            saved = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if result["engine"]["prompt_tokens_computed"] != usage["prompt_tokens"] - saved:
                raise RuntimeError("usage and computed prefill disagree")
            if not cache and saved:
                raise RuntimeError("the cold control reused cached tokens")
        prefix = after["prefix_cache"]
        if prefix["entries"] > prefix["slots"]:
            raise RuntimeError("prefix entries exceed the arena")
        return result


def assistant(result):
    # Deliberately preserve whitespace and split reasoning exactly as delivered.
    return {"role": "assistant", "content": result["content"],
            "reasoning_content": result["reasoning_content"]}


def compare(hot, cold):
    for field in ("content", "reasoning_content", "finish"):
        if hot[field] != cold[field]:
            raise RuntimeError(f"cached and cold replies differ in {field}")
    if hot["usage"]["completion_tokens"] != cold["usage"]["completion_tokens"]:
        raise RuntimeError("cached and cold reply token counts differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=default_url())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cancel-deltas", type=int, default=384)
    parser.add_argument("--prompt-lines", type=int, default=96)
    parser.add_argument("--thinking", action="store_true",
                        help="use the template's native thinking mode (also for always-on templates)")
    parser.add_argument("--require-answer-hit", action="store_true",
                        help="require at least one cancelled replay to reuse beyond its original prompt")
    parser.add_argument("--tag", default="cancel-prefix-157")
    args = parser.parse_args()
    if min(args.rounds, args.cancel_deltas, args.prompt_lines) < 1 or args.cancel_deltas >= 4096:
        parser.error("positive sizes required; cancel-deltas must be below 4096")
    probe = Probe(args.url, thinking=args.thinking)
    report = {"model": probe.model, "source_revision": args.source_revision,
              "binary_sha256": args.binary_sha256, "args": vars(args) | {"out": str(args.out)},
              "cases": [], "passed": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for round_id in range(args.rounds):
            for cancel in (False, True):
                tag = f"{args.tag}-{round_id}-{'cancel' if cancel else 'complete'}"
                messages = [{"role": "system", "content": f"{tag}. Follow the user's instructions precisely."},
                            {"role": "user", "content": "Reference notes:\n" + "\n".join(
                                f"Item {i}: the blue observatory records wind, clouds and distant stars."
                                for i in range(args.prompt_lines)) +
                             "\nWrite 500 numbered sentences about an imaginary observatory. Keep writing until all 500 are done."}]
                case = {"tag": tag, "messages": messages, "replays": []}
                report["cases"].append(case)
                first = probe.chat(messages, 4096 if cancel else args.cancel_deltas,
                                   cancel_deltas=args.cancel_deltas if cancel else 0, ignore_eos=True)
                case["first"] = first
                initial_prompt = first["engine"]["prompt_tokens"]
                if first["engine"]["prompt_tokens_computed"] != initial_prompt:
                    raise RuntimeError("the initial probe was not cold; use a fresh --tag")
                if first["after"]["prefix_cache"]["close_entries"] <= first["before"]["prefix_cache"]["close_entries"]:
                    raise RuntimeError("the first answer retained no rolling snapshot")
                history = messages + [assistant(first), {"role": "user", "content":
                    "Now reply with exactly the words: The observatory is ready."}]
                # Test the exact delivered history before another request can
                # replace its rolling entry. The cold control opts out entirely.
                for variant in ("exact", "shortened", "edited_prompt"):
                    submitted = copy.deepcopy(history)
                    if variant == "shortened":
                        field = "content" if submitted[-2]["content"] else "reasoning_content"
                        submitted[-2][field] = submitted[-2][field][:-96]
                    elif variant == "edited_prompt":
                        submitted[0]["content"] = "A completely different system instruction."
                    hot = probe.chat(submitted, 32)
                    cold = probe.chat(submitted, 32, cache=False)
                    case["replays"].append({"variant": variant, "messages": submitted, "hot": hot, "cold": cold})
                    compare(hot, cold)
                    saved = hot["usage"]["prompt_tokens_details"]["cached_tokens"]
                    if variant == "exact":
                        case["answer_prefix_hit"] = saved > initial_prompt
                        if not saved:
                            raise RuntimeError(f"{tag}: unchanged history lost all prefix reuse")
                    if variant == "edited_prompt" and saved:
                        raise RuntimeError("edited prompt incorrectly attached a cached prefix")
                    print(f"{tag} {variant}: {saved} cached, "
                          f"{hot['engine']['prompt_tokens_computed']} computed; cold reply matches", flush=True)
                    if variant == "exact":
                        continued = submitted + [assistant(hot), {"role": "user", "content":
                            "One more turn: reply with exactly the words Clear skies tonight."}]
                        later = probe.chat(continued, 32)
                        later_cold = probe.chat(continued, 32, cache=False)
                        case["replays"].append({"variant": "following_turn", "messages": continued,
                                                "hot": later, "cold": later_cold})
                        compare(later, later_cold)
                        later_saved = later["usage"]["prompt_tokens_details"]["cached_tokens"]
                        if later_saved < saved:
                            raise RuntimeError("the following conversation turn lost answer-prefix reuse")
                        print(f"{tag} following_turn: {later_saved} cached; cold reply matches", flush=True)
        report["cancelled_answer_hits"] = sum(
            case["first"]["cancelled"] and case["answer_prefix_hit"] for case in report["cases"])
        if args.require_answer_hit and not report["cancelled_answer_hits"]:
            raise RuntimeError("no cancelled answer snapshot matched a cut; inspect the rendered history and arena")
        report["passed"] = True
    finally:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
