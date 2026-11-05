#!/usr/bin/env python3
"""Check interrupted SSE delivery and drain accounting over real loopback HTTP."""
from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from serve_cancel_prefix_check import Probe, assistant, compare
import serve_cancel_prefix_check as check


@contextmanager
def fixture(always_on=False):
    state = {"scheduler": {"active": 0, "queued": 0, "prefilling": 0, "prompts_prefilled": 0,
                           "prompt_tokens": 0, "prompt_tokens_computed": 0, "tokens_generated": 0,
                           "prefill_ms": 0, "decode_steps": 0},
             "service": {"requests_total": 0, "requests_cancelled": 0, "requests_shed_pool": 0},
             "prefix_cache": {"entries": 1, "slots": 2}}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            with lock:
                payload = {"data": [{"id": "fixture"}]} if self.path == "/v1/models" else state
                data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if always_on and "enable_thinking" in payload["chat_template_kwargs"]:
                self.send_error(400, "this template has no enable_thinking knob")
                return
            count = payload["max_tokens"]
            cached = 8 if payload["prefix_cache"] else 0
            with lock:
                state["service"]["requests_total"] += 1
                state["scheduler"]["active"] = 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            cancelled = False
            try:
                for i in range(count):
                    # Unicode and whitespace are significant in a rerendered
                    # prompt. Fragment the SSE bytes across writes too.
                    field = "reasoning_content" if i == 0 else "content"
                    event = {"id": "chatcmpl-fixture", "choices": [{"delta": {field: " α\n"}}]}
                    data = ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode()
                    self.wfile.write(data[:7])
                    self.wfile.flush()
                    self.wfile.write(data[7:])
                    self.wfile.flush()
                    time.sleep(.01)
                event = {"choices": [{"delta": {}, "finish_reason": "length"}],
                         "usage": {"prompt_tokens": 20, "completion_tokens": count,
                                   "prompt_tokens_details": {"cached_tokens": cached}}}
                self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                cancelled = True
            finally:
                # Published scheduler metrics may lag the client's last event.
                time.sleep(.03)
                with lock:
                    sched = state["scheduler"]
                    sched["active"] = 0
                    sched["prompts_prefilled"] += 1
                    sched["prompt_tokens"] += 20
                    sched["prompt_tokens_computed"] += 20 - cached
                    sched["tokens_generated"] += i + 1
                    sched["prefill_ms"] += 1
                    sched["decode_steps"] += i
                    state["service"]["requests_cancelled"] += cancelled

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield Probe(f"http://127.0.0.1:{server.server_port}", timeout=3)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class PrefixProbe:
    model = "fixture"
    rolling = True
    saved_tokens = 1500

    def __init__(self, *args, **kwargs):
        self.prefixes = set()
        self.close_entries = 0

    def chat(self, messages, max_tokens, *, cache=True, cancel_deltas=0, ignore_eos=False):
        # The cache legitimately shares the system and user prefix
        # even when the delivered assistant answers differ.
        key = json.dumps(messages[:2], sort_keys=True)
        saved = self.saved_tokens if cache and key in self.prefixes else 0
        before = self.close_entries
        if cache:
            self.prefixes.add(key)
            self.close_entries += int(self.rolling)
        return {"content": "Observation " * 64, "reasoning_content": "",
                "finish": "length", "cancelled": bool(cancel_deltas),
                "usage": {"prompt_tokens": 2000, "completion_tokens": 128,
                          "prompt_tokens_details": {"cached_tokens": saved}},
                "engine": {"prompt_tokens": 2000, "prompt_tokens_computed": 2000 - saved},
                "before": {"prefix_cache": {"close_entries": before}},
                "after": {"prefix_cache": {"close_entries": self.close_entries}}}

class CancellationProbeTest(unittest.TestCase):
    def test_edited_controls_stay_cold_across_completed_and_cancelled_rounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            argv = ["serve_cancel_prefix_check.py", "--url", "http://fixture:18081",
                    "--out", str(report_path), "--source-revision", "fixture",
                    "--binary-sha256", "fixture", "--rounds", "2", "--thinking"]
            with patch.object(check, "Probe", PrefixProbe), patch.object(sys, "argv", argv), \
                    patch.object(check, "default_url", side_effect=AssertionError("explicit URL needs no site config")), \
                    redirect_stdout(io.StringIO()):
                check.main()
            report = json.loads(report_path.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 4)
        edited = [replay for case in report["cases"] for replay in case["replays"]
                  if replay["variant"] == "edited_prompt"]
        self.assertEqual(len({json.dumps(replay["messages"][:2]) for replay in edited}), 4)
        self.assertTrue(all(replay["hot"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
                            for replay in edited))

    def test_prompt_only_cache_requires_reuse_without_generated_state(self):
        for rolling, saved_tokens, error in ((False, 1500, None),
                                            (True, 1500, "generated-state snapshot"),
                                            (False, 2500, "reused generated state"),
                                            (False, 0, "lost all prefix reuse")):
            with self.subTest(rolling=rolling, saved_tokens=saved_tokens), \
                    tempfile.TemporaryDirectory() as temporary:
                report_path = Path(temporary) / "report.json"
                argv = ["serve_cancel_prefix_check.py", "--url", "http://fixture:18081",
                        "--out", str(report_path), "--source-revision", "fixture",
                        "--binary-sha256", "fixture", "--rounds", "1", "--prompt-only-cache"]
                with patch.object(check, "Probe", PrefixProbe), patch.object(sys, "argv", argv), \
                        patch.object(PrefixProbe, "rolling", rolling), \
                        patch.object(PrefixProbe, "saved_tokens", saved_tokens), \
                        redirect_stdout(io.StringIO()):
                    if error:
                        with self.assertRaisesRegex(RuntimeError, error):
                            check.main()
                    else:
                        check.main()
                        report = json.loads(report_path.read_text())
                        self.assertTrue(report["passed"])
                        self.assertEqual(report["cancelled_answer_hits"], 0)
                        self.assertEqual(sum(len(case["replays"]) for case in report["cases"]), 8)

    def test_disconnect_preserves_exact_delivered_history_and_waits_for_retirement(self):
        with fixture() as probe:
            result = probe.chat([{"role": "user", "content": "hello"}], 40, cancel_deltas=3)
            self.assertTrue(result["cancelled"])
            self.assertEqual(assistant(result), {"role": "assistant", "content": " α\n α\n",
                                                 "reasoning_content": " α\n"})
            self.assertEqual(result["after"]["service"]["requests_cancelled"], 1)
            self.assertEqual(result["after"]["scheduler"]["active"], 0)
            self.assertEqual(result["engine"]["prompt_tokens_computed"], 12)

    def test_cold_replay_opts_out_and_uses_published_usage(self):
        with fixture() as probe:
            messages = [{"role": "user", "content": "hello"}]
            hot = probe.chat(messages, 3)
            cold = probe.chat(messages, 3, cache=False)
            compare(hot, cold)
            self.assertEqual(hot["engine"]["prompt_tokens_computed"], 12)
            self.assertEqual(cold["engine"]["prompt_tokens_computed"], 20)
            self.assertEqual(cold["after"]["service"]["requests_total"], 2)

    def test_early_completion_is_not_reported_as_cancellation(self):
        with fixture() as probe:
            with self.assertRaisesRegex(RuntimeError, "finished before"):
                probe.chat([{"role": "user", "content": "hello"}], 1, cancel_deltas=3)

    def test_native_thinking_does_not_send_a_knob_always_on_templates_reject(self):
        with fixture(always_on=True) as probe:
            probe.thinking = True
            result = probe.chat([{"role": "user", "content": "hello"}], 3)
            self.assertEqual(result["reasoning_content"], " α\n")


if __name__ == "__main__":
    unittest.main()
