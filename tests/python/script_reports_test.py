#!/usr/bin/env python3
"""Offline regression tests for the Python helpers called by Bash scripts."""
import hashlib
import io
import json
import math
import random
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import admission_report
import failure_report
import forward_check_report
import load_phase_report
import node_probe_report
import prefill_report
import roce_report
import run_helpers
import serve_pace
import serve_response_report
import serve_streams
import width_sweep_collect
import bench_stream
import bench_compare
import serve_load
import serve_prefill_probe
import fabric_logprob


def capture(function, *args):
    with redirect_stdout(io.StringIO()) as output:
        result = function(*args)
    return result, output.getvalue()


def response(request="chatcmpl-a", content="answer", tokens=4):
    return {"id": request, "choices": [{"finish_reason": "stop", "message": {
        "content": content, "reasoning_content": "think"}}],
        "usage": {"completion_tokens": tokens}}


def chunk(delta, finish=None):
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def pace_log():
    """One MTP request and one short request sharing a reused slot."""
    entries = [
        (0, "INFO", "sched: request 'chatcmpl-a' admitted to slot 0 — first token 1"),
        (0, "DEBUG", "sched: request 'chatcmpl-a' step 1: token 1"),
        (20, "DEBUG", "bus: graph window timeline: replay"),
        (20, "DEBUG", "sched: request 'chatcmpl-a' step 2: token 2"),
        (21, "DEBUG", "sched: request 'chatcmpl-a' step 3: token 3"),
        (60, "DEBUG", "bus: graph window timeline: replay"),
        (60, "DEBUG", "sched: request 'chatcmpl-a' step 4: token 4"),
        (61, "INFO", "engine: slot 0 closed: 3 sampled decode steps, 1 fallbacks"),
        (62, "INFO", "sched: request 'chatcmpl-a' retired (stop)"),
        (80, "INFO", "sched: request 'chatcmpl-b' admitted to slot 0 — first token 1"),
        (80, "DEBUG", "sched: request 'chatcmpl-b' step 1: token 1"),
        (81, "INFO", "engine: slot 0 closed: 0 sampled decode steps, 0 fallbacks"),
        (82, "INFO", "sched: request 'chatcmpl-b' retired (stop)"),
    ]
    return "".join(f"2026-09-11 12:00:00.{ms:03d} {level}  {text}\n"
                   for ms, level, text in entries)


class ScriptReportsTest(unittest.TestCase):
    def test_prefill_logprob_merges_local_argmax_and_logsumexp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "r0.log").write_text(
                "[prefill_tf] rank 0 step 0: target 9 argmax 2 lmax 1.5 lse 2e+0 target_logit nan\n"
                "[prefill_tf_end] rank 0 world 2 positions 1\n")
            (root / "r1.log").write_text(
                "[prefill_tf] rank 1 step 0: target 9 argmax 9 lmax 3 lse 3.5 target_logit 3\n"
                "[prefill_tf_end] rank 1 world 2 positions 1\n")
            rows, ranks = fabric_logprob.load_run(directory, prefill=True)
            self.assertEqual(ranks, [0, 1])
            self.assertEqual(rows[0]["argmax"], 9)
            lp = fabric_logprob.logprobs(rows, ranks)
            self.assertAlmostEqual(lp[0][0], 3 - math.log(math.exp(2) + math.exp(3.5)))
            self.assertTrue(lp[0][1])
            # Equal logits choose the lower global token ID.
            (root / "r0.log").write_text(
                "[prefill_tf] rank 0 step 0: target 9 argmax 2 lmax 3 lse 3.5 target_logit nan\n"
                "[prefill_tf_end] rank 0 world 2 positions 1\n")
            self.assertEqual(fabric_logprob.load_run(directory, prefill=True)[0][0]["argmax"], 2)

    def test_prefill_logprob_rejects_incomplete_or_conflicting_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "r0.log").write_text(
                "[prefill_tf] rank 0 step 0: target 9 argmax 2 lmax 1 lse 2 target_logit nan\n"
                "[prefill_tf_end] rank 0 world 2 positions 1\n")
            for bad in ["", "[prefill_tf] rank 1 step 1: target 9 argmax 9 lmax 3 lse 4 target_logit 3\n",
                        "[prefill_tf] rank 1 step 0: target 8 argmax 9 lmax 3 lse 4 target_logit 3\n",
                        "[prefill_tf] rank 1 step 0: target 9 argmax 9 lmax 3 lse 4 target_logit nan\n"]:
                (root / "r1.log").write_text(bad + "[prefill_tf_end] rank 1 world 2 positions 1\n")
                with self.assertRaises(SystemExit):
                    fabric_logprob.load_run(directory, prefill=True)

    def test_prefill_logprob_rejects_missing_world_and_common_truncation(self):
        row = "[prefill_tf] rank 0 step 0: target 9 argmax 9 lmax 3 lse 4 target_logit 3\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r0.log"
            for tail in ["", "[prefill_tf_end] rank 0 world 2 positions 1\n",
                         "[prefill_tf_end] rank 0 world 1 positions 2\n",
                         row + "[prefill_tf_end] rank 0 world 1 positions 1\n"]:
                path.write_text(row + tail)
                with self.assertRaises(SystemExit):
                    fabric_logprob.load_run(directory, prefill=True)

    def test_prefill_probe_requires_cold_isolated_metric_deltas(self):
        before = {"prompts_prefilled": 2, "prompt_tokens_computed": 100, "prefill_ms": 80.0}
        after = {"prompts_prefilled": 3, "prompt_tokens_computed": 120, "prefill_ms": 90.0}
        response = {"usage": {"prompt_tokens": 20}, "ttft_ms": None}
        result = serve_prefill_probe.measured_prefill(before, after, response)
        self.assertEqual((result["computed_tokens"], result["prefill_ms_per_token"]), (20, 0.5))
        self.assertIsNone(result["ttft_ms"])
        with self.assertRaisesRegex(RuntimeError, "concurrent traffic"):
            serve_prefill_probe.measured_prefill(before, {**after, "prompts_prefilled": 4}, response)
        with self.assertRaisesRegex(RuntimeError, "concurrent traffic"):
            serve_prefill_probe.measured_prefill(before, {**after, "prompt_tokens_computed": 121}, response)
        with self.assertRaisesRegex(RuntimeError, "attached"):
            serve_prefill_probe.measured_prefill(before, {**after, "prompt_tokens_computed": 116},
                {"usage": {"prompt_tokens": 20, "prompt_tokens_details": {"cached_tokens": 4}}})

    def test_prefill_probe_waits_for_retired_metrics_but_rejects_busy_start(self):
        old = {"active": 0, "queued": 0, "prompts_prefilled": 2}
        busy = {"active": 1, "queued": 0, "prompts_prefilled": 2}
        done = {"active": 0, "queued": 0, "prompts_prefilled": 3}
        with patch.object(serve_prefill_probe, "get", side_effect=[{"scheduler": x} for x in [old, busy, done]]), \
                patch.object(serve_prefill_probe.time, "sleep"):
            self.assertEqual(serve_prefill_probe.idle_metrics("host", 1, timeout=5, completed_after=2), done)
        with patch.object(serve_prefill_probe, "get", return_value={"scheduler": busy}):
            with self.assertRaises(RuntimeError):
                serve_prefill_probe.idle_metrics("host", 1)

    def test_prefill_probe_prompts_are_reproducible_and_do_not_silently_truncate(self):
        words = [f"word{i}" for i in range(100)]
        prompt = serve_prefill_probe.prompt_of(words, 40, "trial", random.Random(7))
        self.assertEqual(prompt, serve_prefill_probe.prompt_of(words, 40, "trial", random.Random(7)))
        self.assertEqual(len(prompt.split("\n\n")[1].split()), 40)
        with self.assertRaisesRegex(ValueError, "dataset"):
            serve_prefill_probe.prompt_of(words, 101, "trial", random.Random(7))

    def test_live_stream_uses_usage_and_ignores_role_events(self):
        class Response:
            status = 200
            def __init__(self):
                self.parts = iter([
                    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r',
                    b'\ndata: {"choices":[{"delta":{"content":"two tokens"}}]}\n\n',
                    b'data: {"choices":[{"delta":{"reasoning_content":"think","content":"answer"}}]}\n\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"completion_tokens":7}}\n\ndata: [DONE]\n\n',
                    b''])
            def read1(self, _): return next(self.parts)
        ticks = iter([1.0, 2.0, 4.0, 8.0])
        result = bench_stream.read_completion(Response(), lambda: next(ticks))
        self.assertEqual((result['first'], result['last'], result['tokens'], result['chunks']), (2, 4, 7, 2))
        self.assertEqual(result['text'], 'two tokensthinkanswer')
        self.assertEqual(result['update_gaps_ms'], [2000])

    def test_live_stream_rejects_missing_usage_and_truncation(self):
        class Response:
            status = 200
            def __init__(self, payload): self.parts = iter([payload, b''])
            def read1(self, _): return next(self.parts)
        for payload in [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
                        b'data: {"choices":[{"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                        b'data: {"error":{"message":"failed"}}\n\n']:
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                bench_stream.read_completion(Response(payload))

    def test_phase_rates_have_separate_wall_and_legacy_scopes(self):
        results = [{'tokens': 10, 'first': 2, 'last': 4}, {'tokens': 20, 'first': 3, 'last': 6}]
        metrics = bench_stream.phase_metrics(results, 1, 7)
        self.assertEqual(metrics['wall_tokens_per_s'], 5)
        self.assertEqual(metrics['legacy_output_span_tokens_per_s'], 7.5)
        with self.assertRaises(RuntimeError):
            bench_stream.phase_metrics([{'error': 'connection failed'}], 1, 7)

    def test_high_concurrency_corpus_is_distinct_and_keeps_anchors(self):
        serve_load.check_corpus()
        for prompts in [serve_load.PROMPTS, *serve_load.CLASSES.values()]:
            self.assertEqual(len(prompts), 16)
            self.assertEqual(len(set(prompts)), 16)

    def test_load_reports_actual_thinking_mode_when_switch_is_unsupported(self):
        first, second = Mock(), Mock()
        first.getresponse.return_value = Mock(status=400, read=Mock(return_value=b'enable_thinking is unsupported'))
        second.getresponse.return_value = Mock(status=200)
        out = {}
        with patch.object(serve_load.http.client, 'HTTPConnection', side_effect=[first, second]), \
                patch.object(serve_load, 'read_completion', return_value={'tokens': 1, 'text': 'answer'}):
            serve_load._stream_one('host', 1, 'model', 'prompt', 1, False, out, 0)
        self.assertEqual(out['thinking'], 'server_default')
        self.assertTrue(out['thinking_switch_unsupported'])
        self.assertNotIn('chat_template_kwargs', json.loads(second.request.call_args.kwargs['body']))
        first.close.assert_called_once()
        second.close.assert_called_once()

    def test_load_does_not_retry_an_unrelated_bad_request(self):
        connection = Mock()
        connection.getresponse.return_value = Mock(status=400, read=Mock(return_value=b'invalid max_tokens'))
        with patch.object(serve_load.http.client, 'HTTPConnection', return_value=connection) as factory:
            with self.assertRaisesRegex(RuntimeError, 'invalid max_tokens'):
                serve_load._stream_one('host', 1, 'model', 'prompt', 1, False, {}, 0)
        factory.assert_called_once()
        connection.close.assert_called_once()

    def test_c1_comparison_rejects_regressions_and_unmatched_work(self):
        report = {"schema_version": 1, "model": "fixture", "max_tokens": 320,
                  "temperature": 0, "thinking": False, "phases": []}
        for name in bench_compare.CLASSES:
            for rate in (99, 100, 101):
                report["phases"].append({"class": name, "concurrency": 1,
                    "requests": [{"prompt_sha256": name, "tokens": 320, "finish": "length",
                                  "text": name, "status": 200}],
                    "metrics": {metric: rate for metric in bench_compare.METRICS}})
        self.assertFalse(bench_compare.compare(report, report)["failures"])
        candidate = json.loads(json.dumps(report))
        for phase in candidate["phases"]:
            phase["metrics"]["wall_tokens_per_s"] *= 0.9
        self.assertEqual(len(bench_compare.compare(report, candidate)["failures"]), 5)
        candidate["phases"][0]["requests"][0]["text"] = "different"
        with self.assertRaisesRegex(ValueError, "unmatched"):
            bench_compare.compare(report, candidate)
        with self.assertRaisesRegex(ValueError, "repeats"):
            bench_compare.compare(report, report, min_repeats=4)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dgpp reports ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write(self, name, text):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, name, value):
        return self.write(name, json.dumps(value))

    def stream(self, *events):
        return self.write("stream.sse", ": keepalive\nevent: message\n\n" + "".join(
            "data: " + ("[DONE]" if event is None else json.dumps(event)) + "\n\n"
            for event in events))

    def cli(self, module, *args):
        # An unrelated cwd and paths with spaces exercise the shell-call interface.
        return subprocess.run([sys.executable, str(SCRIPTS / module), *map(str, args)],
                              cwd=self.directory, text=True, capture_output=True)

    def test_prefill_prompt_and_summary(self):
        path = self.write("ids.txt", "1,2\n3,,\n")
        self.assertEqual(prefill_report.repeat_prompt(path, 5), "1,2,3,1,2")
        self.assertEqual(len(prefill_report.repeat_prompt(path, 100).split(",")), 60)
        self.assertEqual(prefill_report.repeat_prompt(path, 0), "")
        log = self.write("prefill.log", "prefill: 4 tokens in 100ms\nprefill: 8 tokens in 200ms\n")
        self.assertEqual(capture(prefill_report.report, log, 4)[1],
                         "      4 tokens:     100 ms prefill =   25.0 ms/token\n")
        self.assertEqual(capture(prefill_report.report, path, 4)[1], "  n4: no prefill line\n")

    def forward_logs(self, second_digest="0123456789abcdef"):
        self.write("w2_r0.log", "digest 0123456789abcdef\nargmax ids: 9,4\nargmax logits: 1,2\n")
        self.write("w2_r1.log", f"digest {second_digest}\nargmax ids: 3,7\nargmax logits: 1,3\n")

    def test_forward_tie_break_and_digest_verdict(self):
        self.forward_logs()
        status, output = capture(forward_check_report.check_forward, self.directory, 2)
        self.assertEqual(status, 0)
        self.assertEqual(output, "cross-rank digests identical: True (1 digests)\nmerged argmax ids: 3,7\n")
        self.forward_logs("fedcba9876543210")
        result = self.cli("forward_check_report.py", self.directory, 2)
        self.assertEqual(result.returncode, 1)
        self.assertIn("identical: False", result.stdout)

    def test_forward_missing_argmax_and_empty_digests_fail(self):
        self.forward_logs()
        self.write("w2_r1.log", "digest 0123456789abcdef\n")
        self.assertEqual(capture(forward_check_report.check_forward, self.directory, 2),
                         (1, "rank 1 has no argmax lines\n"))
        for rank in range(2):
            self.write(f"w2_r{rank}.log", "argmax ids: 1\nargmax logits: 1\n")
        self.assertEqual(capture(forward_check_report.check_forward, self.directory, 2)[0], 1)

    def test_stream_text_order_and_truncated_tail(self):
        path = self.stream(chunk({"reasoning_content": "résumé", "content": " answer"}), None)
        with path.open("ab") as stream:
            stream.write(b'data: {"truncated":\xff\n')
        self.assertEqual(serve_streams.committed_text(path), "résumé answer")
        result = self.cli("serve_streams.py", "text", path)
        self.assertEqual((result.returncode, result.stdout), (0, "résumé answer"))

    def test_normal_stream_reports_reject_malformed_json(self):
        path = self.write("bad.sse", 'data: {"truncated":\n')
        with self.assertRaises(json.JSONDecodeError):
            list(serve_streams.iter_events(path))
        self.assertNotEqual(self.cli("serve_streams.py", "summary", path).returncode, 0)

    def test_stream_summary_counts_usage_but_not_error_or_done(self):
        path = self.stream(chunk({"content": "ok"}, "stop"),
                           {"choices": [], "usage": {"completion_tokens": 1}},
                           {"error": {"message": "shutdown"}}, None)
        self.assertEqual(capture(serve_streams.summary, path)[1],
                         "stream: 2 chunks, error={'message': 'shutdown'}, done=True, finish_reason=stop\n")

    def test_raw_post_error_token_check_and_exit_status(self):
        for tail, status in [("", 0), ('data: {"content":"late', 1),
                             ('data: {"reasoning_content":"late', 1)]:
            with self.subTest(tail=tail):
                path = self.write("error.sse", 'data: {"content":"before"}\ndata: {"error":"gone"}\n' + tail)
                self.assertEqual(self.cli("serve_streams.py", "no-tokens-after-error", path).returncode, status)
        path = self.write("ok.sse", 'data: {"content":"ok"}\n')
        self.assertTrue(serve_streams.no_tokens_after_error(path))

    def test_stream_tool_fragments_and_run_lengths(self):
        path = self.stream(
            chunk({"content": ""}), chunk({"reasoning_content": "think"}),
            chunk({"reasoning_content": " more"}),
            chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {
                "name": "weather", "arguments": '{"city":'}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Oslo"}'}}]}),
            chunk({}, "tool_calls"), {"choices": [], "usage": {"completion_tokens": 5}}, None)
        output = capture(serve_streams.tools_summary, path)[1]
        self.assertIn("chunks: role Rx2 Tstart Targs final usage DONE\n", output)
        self.assertIn("finish: tool_calls usage: {'completion_tokens': 5}\n", output)
        self.assertIn("reasoning: 'think more'\ncontent: ''\ncall 0 call_1 weather {\"city\":\"Oslo\"}\n", output)

    def test_response_schema_and_boolean_population(self):
        city = {"city": "Oslo", "country": "Norway", "population": 1, "landmarks": ["Harbor"]}
        path = self.write_json("json_schema.json", response(content=json.dumps(city)))
        self.assertIn("schema: CONFORMS\n", capture(serve_response_report.report, path)[1])
        city["population"] = True
        self.write_json(path.name, response(content=json.dumps(city)))
        self.assertIn("schema: VIOLATES\n", capture(serve_response_report.report, path)[1])

    def test_response_error_and_non_json_body_remain_informational(self):
        path = self.write_json("error.json", {"error": "overloaded"})
        self.assertEqual(capture(serve_response_report.report, path)[1], "ERROR overloaded\n")
        path = self.write("bad.json", "not json")
        result = self.cli("serve_response_report.py", path)
        self.assertEqual((result.returncode, result.stdout), (0, "not json\n"))

    def test_request_body_escapes_prompt(self):
        prompt = 'A "quoted" prompt\nwith unicode: café'
        result = self.cli("failure_report.py", "request-body", "true", 42, prompt, "test/model")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body, failure_report.request_body(True, 42, prompt, "test/model"))
        self.assertEqual(body["messages"][0]["content"], prompt)
        self.assertIs(body["stream"], True)
        self.assertEqual(body["max_tokens"], 42)

    def test_replay_prefix_must_be_nonempty(self):
        answer = self.write("answer.txt", "résumé answer")
        for prefix, expected in [("résumé", 0), ("other", 1), ("", 1)]:
            with self.subTest(prefix=prefix):
                committed = self.write("prefix.txt", prefix)
                self.assertEqual(self.cli("failure_report.py", "prefix", committed, answer).returncode, expected)

    def test_ops_compare_only_surviving_common_prefix(self):
        self.write("serve_rank0.ops", "a\nb\nc\n")
        self.write("serve_rank1.ops", "a\nb\n")
        self.write("serve_rank2.ops", "killed\n")
        self.write("serve_rank3.ops", "")
        self.assertEqual(capture(failure_report.report_ops, self.directory, 2)[1],
                         "  op streams agree over the first 2 lines\n")
        self.write("serve_rank1.ops", "a\nx\n")
        self.assertIn("FAIL: op streams disagree", capture(failure_report.report_ops, self.directory, 2)[1])
        self.write("serve_rank1.ops", "")
        self.assertIn("fewer than two ops files", capture(failure_report.report_ops, self.directory, 2)[1])

    def test_roce_snapshot_deltas(self):
        before = self.write("before.txt", "header\nn1 d1 xmit_data=0 retry=3 cnp=0\nn2 d1 xmit_data=0\n")
        after = self.write("after.txt", "n1 d1 xmit_data=262144 retry=5 cnp=0 discard=1\nn2 d1 xmit_data=0\nn3 d1 retry=1\n")
        self.assertEqual(capture(roce_report.report, before, after)[1],
                         "n1 d1: xmit      1.0 MiB  retry=2  discard=1\n"
                         "n2 d1: xmit      0.0 MiB  no retransmits, no CNPs, no discards\n")

    def test_node_probe_sums_and_masks(self):
        path = self.write("node.log", "header\n12:00:00 1 2 3 4 5 6 | throttle=0x0\n"
                          "12:00:01 2 3 4 5 6 7 | throttle=0x1\n")
        self.assertEqual(capture(node_probe_report.report, path)[1],
                         "  node.log: 2 samples, allocstall 3, pgmajfault 5, swap 16, "
                         "pgscan_direct 11, compact_stall 13, throttle masks ['0x0', '0x1']\n")

    def test_load_phase_bounds_and_prefill_stripping(self):
        self.write("load.log", "== concurrency 2: 12:00:00 .. 12:00:10\n")
        self.write("world/serve_r0.log",
                   "2026-09-11 12:00:05 INFO  stats: rank 0 | 5 s | good | prefill ignored\n"
                   "2026-09-11 12:00:15 INFO  stats: rank 0 | 5 s | outside\n")
        self.assertEqual(capture(load_phase_report.report, self.directory)[1],
                         "-- concurrency 2 (12:00:00 .. 12:00:10)\n   good\n")

    def test_admission_outcomes_and_metrics(self):
        self.write_json("req_a.json", response())
        self.write_json("req_b.json", {"error": "full"})
        self.write_json("metrics.json", {"service": {"admission": "grow", "reservations_grown": 2,
                        "requests_shed_pool": 1}, "scheduler": {"pool_blocks_in_use": 3,
                        "pool_blocks_total": 4, "tokens_generated": 5}})
        self.assertEqual(capture(admission_report.report, self.directory, 1.0, 3.25, "grow")[1],
                         "  req_a.json: finish stop, completion_tokens 4: 'answer'\n"
                         "  req_b.json: ERROR full\n"
                         "  metrics: admission grow, reservations_grown 2, requests_shed_pool 1, pool 3/4 blocks, tokens 5\n"
                         "  batch wall time under grow: 2.2 s\n")

    def test_pace_measurements_and_wave_report(self):
        path = self.write("pace.log", pace_log())
        first, short = serve_pace.request_paces(path)
        self.assertEqual((first["tokens"], first["replays"], first["tokens_per_replay"]), (4, 2, 1.5))
        self.assertAlmostEqual(first["ms_per_token"], 20, places=3)
        self.assertAlmostEqual(first["ms_per_replay"], 40, places=3)
        self.assertEqual(short, {"request": "chatcmpl-b", "tokens": 1, "short": True})
        result = self.cli("serve_pace.py", path, "--waves")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("(too short)", result.stdout)
        self.assertIn("tok/s", result.stdout)
        self.assertIn("40.000", result.stdout)

    def test_two_token_pace_has_nan_timing(self):
        path = self.write("short.log", "\n".join(pace_log().splitlines()[:4]))
        row = serve_pace.request_paces(path)[0]
        self.assertEqual((row["tokens"], row["replays"]), (2, 1))
        self.assertTrue(math.isnan(row["ms_per_token"]))

    def test_width_collection_reused_slot_and_short_request(self):
        self.write("serve/serve_r0.log", pace_log())
        self.write_json("req_greedy_0.json", response())
        self.write_json("req_greedy_1.json", response("chatcmpl-b", tokens=1))
        self.write_json("req_error.json", {"error": "full"})
        self.write("req_broken.json", "not json")
        tsv = self.write("results.tsv", "header\n")
        result = self.cli("width_sweep_collect.py", self.directory, 64, "mtp", tsv)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("broken: unreadable", result.stdout)
        self.assertIn("error: ERROR full", result.stdout)
        header, first, second = tsv.read_text().splitlines()
        self.assertEqual(header, "header")
        digest = hashlib.sha256(b"think\x00answer").hexdigest()[:12]
        self.assertEqual(first.split("\t"), ["64", "mtp", "greedy_0", "greedy", "4", "stop",
                         "3", "1", "2", "1.5", "40.0", "20.0", digest])
        self.assertEqual(second.split("\t")[6:12], ["0", "0", "-1", "nan", "nan", "nan"])

    def test_process_helpers_without_network_or_sleep(self):
        with patch.object(run_helpers.socket, "socket") as socket:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 0
            self.assertTrue(run_helpers.port_open(9999))
            socket.return_value.__enter__.return_value.connect_ex.assert_called_once_with(("127.0.0.1", 9999))
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            self.assertFalse(run_helpers.port_open(9999))
        with patch.object(sys, "argv", ["run_helpers.py", "random-delay"]), \
                patch.object(run_helpers.random, "uniform", return_value=0.456):
            self.assertEqual(capture(run_helpers.main)[1], "0.46\n")
        with patch.object(sys, "argv", ["run_helpers.py", "elapsed", "100"]), \
                patch.object(run_helpers.time, "time", return_value=102.125):
            self.assertEqual(capture(run_helpers.main)[1], "2.12\n")

    def test_all_module_clis_have_help(self):
        for module in ("admission_report", "failure_report", "forward_check_report",
                       "load_phase_report", "node_probe_report", "prefill_report",
                       "roce_report", "run_helpers", "serve_response_report",
                       "serve_streams", "width_sweep_collect"):
            with self.subTest(module=module):
                result = self.cli(module + ".py", "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_bash_scripts_have_no_inline_python(self):
        for path in SCRIPTS.iterdir():
            if not path.is_file():
                continue
            text = path.read_text()
            if not re.match(r"#!.*\b(?:ba)?sh\b", text):
                continue
            with self.subTest(script=path.name):
                self.assertNotRegex(text, r"\bpython[\d.]*\s+(?:-[cu]\b|-(?=\s|$)|<<)")
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
