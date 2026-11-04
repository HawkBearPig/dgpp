#!/usr/bin/env python3
"""Check interrupted SSE delivery and drain accounting over real loopback HTTP."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from serve_cancel_prefix_check import Probe, assistant, compare


@contextmanager
def fixture():
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


class CancellationProbeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
