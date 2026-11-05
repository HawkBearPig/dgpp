"""Exercise the server's settings handshake before model loading or GPU use."""
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import tempfile
import unittest


class ServeStartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("DGPP_SERVE_TEST_BINARY")
        if not binary:
            raise unittest.SkipTest("set DGPP_SERVE_TEST_BINARY to the freshly built server")
        cls.binary = str(Path(binary).resolve(strict=True))

    def test_tokenizer_cache_config_rejected_before_loading(self):
        with tempfile.TemporaryDirectory(prefix="dgpp-tokenizer-cache-") as directory:
            config = Path(directory) / "cluster.json"
            for value in (-1, 10001, 1.5, 1.0, True, None, "10"):
                with self.subTest(value=value):
                    config.write_text(json.dumps({"model": "unused/model", "world_size": 1,
                                                  "tokenizer": {"word_cache_entries": value}}))
                    result = subprocess.run([self.binary, "--config", str(config)],
                                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                            text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, timeout=10)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("tokenizer.word_cache_entries", result.stdout)

    def test_stream_flush_config_cli_precedence_and_help(self):
        with tempfile.TemporaryDirectory(prefix="dgpp-stream-flush-") as directory:
            config = Path(directory) / "cluster.json"
            for settings, flags, expected in (
                ({}, [], "pass"),
                ({"stream_flush": "wake"}, [], "wake"),
                ({"stream_flush": "wake"}, ["--stream-flush", "pass"], "pass"),
                ({"stream_flush": "pass"}, ["--stream-flush", "wake"], "wake"),
            ):
                with self.subTest(settings=settings, flags=flags):
                    config.write_text(json.dumps({"model": "unused/model", "nodes": ["127.0.0.1"],
                                                  "http": settings}))
                    result = subprocess.run(
                        [self.binary, "--config", str(config), "--model", "", "--checkpoint-dir", directory,
                         "--max-connections", "0", *flags],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DGPP_LOG_LEVEL": "info"},
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn(f"HTTP stream flush {expected}", result.stdout)
                    self.assertIn("all capacity knobs must be >= 1", result.stdout)
        result = subprocess.run([self.binary, "--help"], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=10)
        self.assertIn("--stream-flush", result.stdout)
        self.assertIn("http.stream_flush", result.stdout)

    def test_stream_flush_invalid_cli_values(self):
        for value in ("", "Wake", "poll", "0", "true"):
            with self.subTest(value=value):
                result = subprocess.run([self.binary, "--stream-flush", value], text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("--stream-flush must be pass or wake", result.stdout)

    def test_sse_ping_interval_config_cli_precedence_and_help(self):
        with tempfile.TemporaryDirectory(prefix="dgpp-sse-ping-") as directory:
            config = Path(directory) / "cluster.json"
            for settings, flags, expected in (
                ({}, [], 30),
                ({"sse_ping_interval": 7}, [], 7),
                ({"sse_ping_interval": -1}, [], -1),
                ({"sse_ping_interval": 7}, ["--sse-ping-interval", "-1"], -1),
                ({"sse_ping_interval": -1}, ["--sse-ping-interval", "2"], 2),
            ):
                with self.subTest(settings=settings, flags=flags):
                    config.write_text(json.dumps({"model": "unused/model", "nodes": ["127.0.0.1"],
                                                  "http": settings}))
                    result = subprocess.run(
                        [self.binary, "--config", str(config), "--model", "", "--checkpoint-dir", directory,
                         "--max-connections", "0", *flags],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DGPP_LOG_LEVEL": "info"},
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn(f"SSE ping interval {expected} s", result.stdout)
                    self.assertIn("all capacity knobs must be >= 1", result.stdout)
        help_result = subprocess.run([self.binary, "--help"], text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, timeout=10)
        self.assertIn("--sse-ping-interval", help_result.stdout)
        self.assertIn("http.sse_ping_interval", help_result.stdout)

    def test_sse_ping_interval_invalid_cli_values(self):
        for value in ("0", "-2", "true", "1.5", "1.0", "2147483648", "1x", ""):
            with self.subTest(value=value):
                result = subprocess.run([self.binary, "--sse-ping-interval", value], text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("--sse-ping-interval must be", result.stdout)

    def test_dflash_switch_and_explicit_disable_precedence(self):
        with tempfile.TemporaryDirectory(prefix="dgpp-dflash-switch-") as directory:
            config = Path(directory) / "cluster.json"
            config.write_text(json.dumps({
                "model": "unused/model", "nodes": ["127.0.0.1"],
                "engine": {"decode_graph": True, "mtp": True, "mtp_depth": 2,
                           "mtp_schedule": True, "lookup_draft": True, "lookup_tail": 2,
                           "draft_head": "screen"},
            }))
            common = [self.binary, "--config", str(config), "--model", "",
                      "--checkpoint-dir", directory, "--max-connections", "0"]
            for flags, contradiction in (
                (["--dflash-model", directory], False),
                (["--mtp", "--dflash-model", directory], True),
                (["--dflash-model", directory, "--mtp"], True),
                (["--no-dflash", "--dflash-model", "unused/not-cached"], False),
                (["--dflash-model", "unused/not-cached", "--no-dflash"], False),
            ):
                with self.subTest(flags=flags):
                    result = subprocess.run(common + flags, text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, timeout=10,
                                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                    self.assertEqual(result.returncode, 2 if contradiction else 1, result.stdout)
                    self.assertIn("enable one or the other" if contradiction else
                                  "all capacity knobs must be >= 1", result.stdout)

    def run_world(self, settings, head_modes=None, nvme_bindings=None):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="dgpp-startup-") as directory:
            # The invalid local HTTP capacity ends every rank after the handshake,
            # before checkpoint access, CUDA initialization or RDMA setup.
            common = [self.binary, "--checkpoint-dir", str(Path(directory) / "unused"),
                      "--world", str(len(settings)), "--journal-port", str(port),
                      "--fabric-port", "0", "--max-connections", "0",
                      "--rendezvous-timeout-ms", "5000"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "DGPP_LOG_LEVEL": "info"}
            processes, files = [], []
            try:
                for rank, (slots, threshold) in enumerate(settings):
                    args = common + ["--rank", str(rank), "--max-concurrency", str(slots)]
                    if nvme_bindings is not None:
                        config = Path(directory) / f"rank{rank}.json"
                        config.write_text(json.dumps({
                            "model": "unused/model", "nodes": ["127.0.0.1"] * len(settings),
                            "nvme_cache": {"enabled": True, "capacity_gib": 1,
                                           "reuse_binding": nvme_bindings[rank]},
                        }))
                        args += ["--config", str(config), "--model", ""]
                    if head_modes is not None and head_modes[rank] is not None:
                        args += ["--fp8-head", head_modes[rank]]
                    if rank:
                        args += ["--peer", "127.0.0.1"]
                    if threshold is not None:
                        args += ["--graph-batch-min-live", str(threshold)]
                    output = open(Path(directory) / f"rank{rank}.log", "w+")
                    files.append(output)
                    processes.append(subprocess.Popen(args, env=env, stdout=output,
                                                      stderr=subprocess.STDOUT))
                logs = []
                for process, output in zip(processes, files):
                    process.wait(timeout=15)
                    output.seek(0)
                    log = output.read()
                    self.assertEqual(process.returncode, 1, log)
                    self.assertIn("all capacity knobs must be >= 1", log)
                    logs.append(log)
                self.assertIn(f"settings pushed to {len(settings) - 1} peer(s)", logs[0])
                head = logs[0].split("peer(s): ", 1)[1].splitlines()[0]
                for log in logs[1:]:
                    marker = " ; runs: " if "settings override" in log else "settings from rank 0: "
                    self.assertIn(marker, log)
                    self.assertEqual(log.split(marker, 1)[1].splitlines()[0], head)
                return logs
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                for output in files:
                    output.close()

    def test_nvme_reuse_binding_is_selected_by_rank_zero(self):
        binding = "0123456789abcdef"
        for bindings in ([binding, ""], ["", binding]):
            with self.subTest(bindings=bindings):
                logs = self.run_world([(4, None), (4, None)], nvme_bindings=bindings)
                head = logs[0].split("peer(s): ", 1)[1].splitlines()[0]
                self.assertEqual(f"nvme_reuse={binding}" in head, bool(bindings[0]))

    def test_memlock_prepared_before_cuda_with_config_debug_and_mlock_off(self):
        _, inherited_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        hard = 8192 if inherited_hard == resource.RLIM_INFINITY else min(inherited_hard, 8192)

        def lower_soft_limit():
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, hard))

        with tempfile.TemporaryDirectory(prefix="dgpp-memlock-") as directory:
            config = Path(directory) / "cluster.json"
            config.write_text(json.dumps({
                "model": "unused/model", "nodes": ["127.0.0.1"],
                "node_env": [{"DGPP_LOG_LEVEL": "debug", "DGPP_MLOCK": "off"}],
            }))
            # Clear the config's model id so reaching CUDA needs no cached checkpoint.
            result = subprocess.run(
                [self.binary, "--config", str(config), "--rank", "0", "--model", "",
                 "--checkpoint-dir", str(Path(directory) / "unused")],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DGPP_LOG_LEVEL": "info"},
                preexec_fn=lower_soft_limit, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=15)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("memlock before preparation: RLIMIT_MEMLOCK soft 0 bytes", result.stdout)
            after = f"memlock after preparation: RLIMIT_MEMLOCK soft {hard} bytes, hard {hard} bytes"
            self.assertIn(after, result.stdout)
            self.assertIn("no CUDA device visible", result.stdout)
            self.assertLess(result.stdout.index(after), result.stdout.index("no CUDA device visible"))

    def test_matching_defaults_are_quiet(self):
        for slots in (1, 2, 4, 8):
            with self.subTest(slots=slots):
                logs = self.run_world([(slots, None), (slots, 0)])
                self.assertNotIn("settings override", logs[1])
                self.assertIn("settings from rank 0:", logs[1])
                self.assertIn(f"batchmin={min(2, slots)} ", logs[1])

    def test_explicit_default_matches_automatic_on_every_peer(self):
        for slots in (1, 4):
            default = min(2, slots)
            for head in (0, default):
                with self.subTest(slots=slots, head=head):
                    logs = self.run_world([(slots, head), (slots, None),
                                           (slots, 0), (slots, default)])
                    for log in logs[1:]:
                        self.assertNotIn("settings override", log)
                        self.assertIn("settings from rank 0:", log)
                        self.assertIn(f"batchmin={default} ", log)

    def test_different_threshold_still_warns_and_adopts_head(self):
        for head, peer, effective in ((0, 1, 2), (3, 0, 3), (3, 4, 3)):
            with self.subTest(head=head, peer=peer):
                log = self.run_world([(4, head), (4, peer)])[1]
                self.assertIn("WARN", log)
                self.assertIn("rank 0's settings override this rank's own", log)
                own, adopted = log.split(" ; runs: ", 1)
                self.assertIn(f"batchmin={peer or 2} ", own)
                self.assertIn(f"batchmin={effective} ", adopted)

    def test_fp8_head_default_and_rank_zero_override(self):
        for head, peer, effective in ((None, None, "gemv"), ("mma", None, "mma"),
                                      ("gemv", "mma", "gemv")):
            with self.subTest(head=head, peer=peer):
                logs = self.run_world([(4, 0), (4, 0)], [head, peer])
                self.assertIn(f"fp8head={effective} ", logs[0])
                if head != peer:
                    own, adopted = logs[1].split(" ; runs: ", 1)
                    self.assertIn(f"fp8head={peer or 'gemv'} ", own)
                    self.assertIn(f"fp8head={effective} ", adopted)
                else:
                    self.assertNotIn("settings override", logs[1])

    def test_invalid_fp8_head_names_setting_before_gpu_use(self):
        result = subprocess.run(
            [self.binary, "--checkpoint-dir", "/unused", "--fp8-head", "auto"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("engine.fp8_head", result.stdout)
        self.assertIn("must be gemv or mma", result.stdout)
        self.assertNotIn("no CUDA device", result.stdout)

    def test_mma_requires_fp8_dense_weights_before_gpu_use(self):
        result = subprocess.run(
            [self.binary, "--checkpoint-dir", "/unused", "--fp8-head", "mma"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("engine.fp8_head mma requires engine.dense_weights fp8", result.stdout)
        self.assertNotIn("no CUDA device", result.stdout)

    def test_different_concurrency_still_warns(self):
        log = self.run_world([(4, 0), (1, 0)])[1]
        self.assertIn("rank 0's settings override this rank's own", log)
        own, adopted = log.split(" ; runs: ", 1)
        self.assertIn("conc=1 ", own)
        self.assertIn("batchmin=1 ", own)
        self.assertIn("conc=4 ", adopted)
        self.assertIn("batchmin=2 ", adopted)


if __name__ == "__main__":
    unittest.main()
