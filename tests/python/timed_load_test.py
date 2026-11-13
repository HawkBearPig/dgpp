#!/usr/bin/env python3
"""Counter publication regressions, with a fake clock and a local SSE fixture."""
from contextlib import ExitStack, contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import timed_load


def snapshot(prompts=10, tokens=100, **changes):
    return {"active": 0, "queued": 0, "prompts_prefilled": prompts,
            "tokens_generated": tokens, "step_ms": 100., "prefill_ms": 50.,
            "decode_steps": 50, "decode_rows": 50, "prompt_tokens": 500,
            "prompt_tokens_computed": 500, **changes}


class ReconciliationTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.get = stack.enter_context(patch.object(timed_load, "get"))
        stack.enter_context(patch.object(timed_load.time, "monotonic", side_effect=lambda: self.now))
        self.sleep = stack.enter_context(patch.object(timed_load.time, "sleep", side_effect=self.advance))
        self.before = snapshot()
        self.requests = [{"tokens": 320}] * 4
        self.done = snapshot(14, 1380, step_ms=2100.)

    def advance(self, seconds):
        self.now += seconds

    def reads(self, *values):
        sequence = iter(values)
        self.get.side_effect = lambda *a, **k: {"scheduler": next(sequence, values[-1])}

    def reconcile(self, **kwargs):
        return timed_load.reconciled_metrics("host", 1, self.before, self.requests, **kwargs)

    def test_ready_snapshot_needs_one_read_and_no_sleep(self):
        self.reads(self.done)
        self.assertEqual(self.reconcile(), self.done)
        self.get.assert_called_once_with("host", 1, "/v1/metrics", timeout=5)
        self.sleep.assert_not_called()

    def test_identical_stale_reads_and_two_token_tail_do_not_establish_readiness(self):
        late = {**self.done, "tokens_generated": 1378, "step_ms": 2090.}
        self.reads(self.before, self.before, late, late, late, self.done)
        self.assertEqual(self.reconcile(), self.done)
        self.assertEqual(self.get.call_count, 6)
        self.assertEqual(self.sleep.call_count, 5)

    def test_both_counters_and_idle_gauges_must_match(self):
        self.reads({**self.done, "prompts_prefilled": 13},
                   {**self.done, "active": 1}, {**self.done, "queued": 1}, self.done)
        self.assertEqual(self.reconcile(), self.done)
        self.assertEqual(self.get.call_count, 4)

    def test_stale_counter_deadline_reports_expected_and_observed_without_real_sleep(self):
        self.reads({**self.done, "tokens_generated": 1378})
        with self.assertRaisesRegex(RuntimeError, "deadline.*1280.*1278.*active/queued"):
            self.reconcile(timeout=.025)
        self.assertEqual(self.get.call_count, 3)
        self.assertAlmostEqual(self.now, .025)
        self.assertAlmostEqual(self.get.call_args.kwargs["timeout"], .005)

    def test_overshoot_and_counter_reset_fail_immediately(self):
        for field, value, reason in (("prompts_prefilled", 15, "unexpected"),
                                     ("tokens_generated", 1381, "unexpected"),
                                     ("tokens_generated", 99, "regressed"),
                                     ("prompts_prefilled", 9, "regressed")):
            with self.subTest(field=field, value=value):
                self.reads({**self.done, field: value})
                with self.assertRaisesRegex(RuntimeError, reason + ".*expected delta.*observed delta"):
                    self.reconcile()
        self.sleep.assert_not_called()

    def test_http_timeout_and_slow_response_cannot_extend_the_deadline(self):
        self.get.side_effect = TimeoutError("socket timed out")
        with self.assertRaisesRegex(RuntimeError, "timed out reading.*1280"):
            self.reconcile()
        def slow(*args, **kwargs):
            self.advance(kwargs["timeout"])
            return {"scheduler": self.done}
        self.get.side_effect = slow
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.reconcile()
        self.sleep.assert_not_called()

    def test_initial_baseline_requires_idle_and_timeout_must_be_valid(self):
        self.reads({**self.before, "active": 1})
        with self.assertRaisesRegex(RuntimeError, "expected an idle scheduler"):
            timed_load.reconciled_metrics("host", 1, timeout=.025)
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reconcile(timeout=value)

    def test_phase_keeps_client_report_and_engine_formula(self):
        report = {"concurrency": 4, "requests": self.requests,
                  "metrics": {"wall_tokens_per_s": 42}, "class": "prose"}
        def phase(*args):
            args[-1].append(report.copy())
            return 42
        self.reads(self.before, {**self.done, "tokens_generated": 1378}, self.done)
        reports = []
        with patch.object(timed_load.serve_load, "phase", side_effect=phase), redirect_stdout(io.StringIO()):
            rate = timed_load.TimedLoad().phase("host", 1, "model", 4, 320, False, 0, reports=reports)
        self.assertEqual(rate, 42)
        self.assertEqual({k: v for k, v in reports[0].items() if k != "engine"}, report)
        self.assertEqual(reports[0]["engine"]["decode_tokens"], 1276)
        self.assertEqual(reports[0]["engine"]["tokens_per_s"], 638)

    def test_foreign_work_between_phases_is_not_silently_absorbed(self):
        load = timed_load.TimedLoad()
        load.previous = self.done
        self.reads(snapshot(15, 1388))
        reports = []
        with patch.object(timed_load.serve_load, "phase") as phase:
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                load.phase("host", 1, "model", 1, 8, False, 0, reports=reports)
        phase.assert_not_called()
        self.assertEqual(reports, [])

    def test_unreconciled_phase_does_not_append_a_result(self):
        def phase(*args):
            args[-1].append({"concurrency": 4, "requests": self.requests})
        self.reads(self.before, snapshot(15, 1381))
        reports = []
        with patch.object(timed_load.serve_load, "phase", side_effect=phase):
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                timed_load.TimedLoad().phase("host", 1, "model", 4, 320, False, 0, reports=reports)
        self.assertEqual(reports, [])

    def test_invalid_engine_time_never_produces_a_rate(self):
        def phase(*args):
            args[-1].append({"concurrency": 4, "requests": self.requests})
        for elapsed in (0, -1, float("nan"), float("inf")):
            with self.subTest(elapsed=elapsed):
                self.reads(self.before, {**self.done, "step_ms": self.before["step_ms"] + elapsed})
                reports = []
                with patch.object(timed_load.serve_load, "phase", side_effect=phase):
                    with self.assertRaisesRegex(RuntimeError, "no measurable decode work"):
                        timed_load.TimedLoad().phase("host", 1, "model", 4, 320, False, 0, reports=reports)
                self.assertEqual(reports, [])


@contextmanager
def serving_fixture(delayed, isolation_change=None, require_default_thinking=False):
    """Publish the final two tokens only on the third metrics read after work."""
    lock = threading.Lock()
    state = {"current": snapshot(), "pending_reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, content_type, body):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                value = {"data": [{"id": "fixture"}]}
            elif self.path == "/v1/metrics":
                with lock:
                    meters = state["current"].copy()
                    if state["pending_reads"]:
                        state["pending_reads"] -= 1
                        meters["tokens_generated"] -= 2
                        meters["step_ms"] -= 4
                    value = {"scheduler": meters}
            else:
                self.send_error(404)
                return
            self.reply("application/json", json.dumps(value).encode())

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if require_default_thinking and "chat_template_kwargs" in request:
                self.send_error(400, "enable_thinking is unsupported")
                return
            tokens = request["max_tokens"]
            text = "fixture answer"
            with lock:
                if isolation_change:
                    text = "answer 雪\n  exact whitespace"
                    if state["requests"]:
                        if isolation_change == "text":
                            text = "answer 雪\n  changed whitespace"
                        elif isolation_change == "prefix":
                            text += " extended"
                        elif isolation_change == "tokens":
                            tokens -= 1
                state["requests"].append(request)
                m = state["current"]
                m["prompts_prefilled"] += 1
                m["tokens_generated"] += tokens
                m["step_ms"] += (tokens - 1) * 2
                m["decode_steps"] += tokens - 1
                m["decode_rows"] += tokens - 1
                m["prefill_ms"] += 3
                m["prompt_tokens"] += 20
                m["prompt_tokens_computed"] += 20
                state["pending_reads"] = 2 if delayed else 0
            events = [
                {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "length"}],
                 "usage": {"prompt_tokens": 20, "completion_tokens": tokens}},
            ]
            body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            self.reply("text/event-stream", (body + "data: [DONE]\n\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
    thread.start()
    try:
        yield server.server_port, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class TimedLoadCliTest(unittest.TestCase):
    def test_mixed_sweep_without_warmup_or_isolation(self):
        with tempfile.TemporaryDirectory() as directory, serving_fixture(delayed=False) as (port, state):
            output = Path(directory) / "mixed.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/timed_load.py"), "127.0.0.1", str(port),
                "--warm", "0", "--concurrency", "1,4", "--max-tokens", "8",
                "--json-out", str(output)], cwd=directory, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertNotIn("isolation", report)
            self.assertEqual(len(state["requests"]), 5)
            self.assertEqual([p["class"] for p in report["phases"]], ["mixed", "mixed"])
            self.assertEqual([p["engine"]["tokens_per_s"] for p in report["phases"]], [500, 500])

    def test_warmup_isolation_repeated_classes_preserve_work(self):
        scripts = [ROOT / "scripts/serve_load.py", ROOT / "scripts/timed_load.py"]
        runs = []
        with tempfile.TemporaryDirectory(prefix="dgpp timed load ") as directory:
            for script in scripts:
                with self.subTest(script=script), serving_fixture(delayed=True) as (port, state):
                    output = Path(directory) / "report.json"
                    result = subprocess.run([
                        sys.executable, str(script), "127.0.0.1", str(port),
                        "--warm", "1", "--isolation", "4", "--concurrency", "1,4",
                        "--classes", "prose,code", "--repeat", "2", "--max-tokens", "8",
                        "--json-out", str(output)], cwd=directory, text=True, capture_output=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads(output.read_text())
                    isolation = report["isolation"]
                    self.assertTrue(isolation["matched"])
                    self.assertTrue(isolation["text_match"])
                    self.assertTrue(isolation["token_count_match"])
                    self.assertIsNone(isolation["first_difference_char"])
                    self.assertEqual(len(isolation["batched"]), 4)
                    self.assertEqual(len(report["phases"]), 8)
                    self.assertEqual(len(state["requests"]), 26)
                    if script.name == "timed_load.py":
                        for phase in report["phases"]:
                            engine = phase["engine"]
                            self.assertEqual(engine["tokens_generated"], 8 * phase["concurrency"])
                            self.assertEqual(engine["decode_tokens"], 7 * phase["concurrency"])
                            self.assertEqual(engine["tokens_per_s"], 500)
                    # Arrival order across worker threads is intentionally unconstrained.
                    runs.append(sorted(json.dumps(r, sort_keys=True) for r in state["requests"]))
            self.assertEqual(runs[0], runs[1])

    def test_isolation_failure_saves_complete_diagnostics_before_exit(self):
        for script in ("serve_load.py", "timed_load.py"):
            for change in ("text", "prefix", "tokens"):
                with self.subTest(script=script, change=change), tempfile.TemporaryDirectory() as directory:
                    fallback = change == "text"
                    with serving_fixture(False, change, fallback) as (port, state):
                        output = Path(directory) / "failed.json"
                        result = subprocess.run([
                            sys.executable, str(ROOT / "scripts" / script), "127.0.0.1", str(port),
                            "--warm", "0", "--isolation", "2", "--concurrency", "1",
                            "--max-tokens", "8", "--temperature", "0.7", "--json-out", str(output)],
                            cwd=directory, text=True, capture_output=True, encoding="utf-8", timeout=20,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.assertIn("transcript isolation failed", result.stderr)
                        report = json.loads(output.read_text())
                        self.assertEqual(report["schema_version"], 1)
                        self.assertEqual(report["phases"], [])
                        self.assertEqual(report["temperature"], .7)
                        self.assertEqual(report["thinking"], "server_default" if fallback else False)
                        self.assertFalse(report["thinking_requested"])
                        self.assertEqual(report["thinking_switch_unsupported"], fallback)
                        self.assertEqual(len(state["requests"]), 3)
                        isolation = report["isolation"]
                        self.assertFalse(isolation["matched"])
                        self.assertEqual(isolation["concurrency"], 2)
                        self.assertEqual(isolation["text_match"], change == "tokens")
                        self.assertEqual(isolation["token_count_match"], change != "tokens")
                        alone = isolation["alone"]
                        batched = isolation["batched"]
                        self.assertEqual(alone["text"], "answer 雪\n  exact whitespace")
                        expected = {"text": "answer 雪\n  changed whitespace",
                                    "prefix": alone["text"] + " extended", "tokens": alone["text"]}[change]
                        self.assertEqual(batched[0]["text"], expected)
                        self.assertEqual(batched[0]["tokens"], 7 if change == "tokens" else 8)
                        self.assertEqual(isolation["first_difference_char"],
                                         {"text": 11, "prefix": len(alone["text"]), "tokens": None}[change])
                        captured = [r["request"] for r in [alone] + batched]
                        self.assertEqual(sorted(map(json.dumps, captured)),
                                         sorted(map(json.dumps, state["requests"])))
                        self.assertTrue(all(r["temperature"] == 0 for r in captured))
                        self.assertEqual("chat_template_kwargs" in captured[0], not fallback)


if __name__ == "__main__":
    unittest.main()
