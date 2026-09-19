"""Exercise the cross-build CLI without Docker, CUDA or target hardware."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SparkCrossTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cross build ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo with spaces"
        (self.repo / "scripts").mkdir(parents=True)
        self.script = self.repo / "scripts/spark-cross"
        shutil.copy2(ROOT / "scripts/spark-cross", self.script)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "calls.jsonl"
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
                    "CROSS_TEST_LOG": str(self.log), "DGPP_CROSS_IMAGE": "test-cross:local",
                    "DGPP_BUILD_JOBS": "3"}
        for name in ("docker", "cmake"):
            stub = self.bin / name
            stub.write_text('''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CROSS_TEST_LOG"], "a") as out:
    out.write(json.dumps([name, *args]) + "\\n")
if name == "docker":
    if args[:2] == ["image", "inspect"]:
        sys.exit(int(os.environ.get("CROSS_TEST_INSPECT_RC", "0")))
    if args[0] == "build":
        sys.exit(int(os.environ.get("CROSS_TEST_IMAGE_RC", "0")))
    if os.environ.get("CROSS_TEST_RUN_BUILD") == "1":
        command = args[args.index("test-cross:local") + 1:]
        sys.exit(subprocess.call(command))
    sys.exit(int(os.environ.get("CROSS_TEST_RUN_RC", "0")))
key = "CROSS_TEST_CONFIGURE_RC" if args[0] == "--preset" else "CROSS_TEST_BUILD_RC"
sys.exit(int(os.environ.get(key, "0")))
''')
            stub.chmod(0o755)

    def run_cli(self, *args, **env):
        return subprocess.run([str(self.script), *args], cwd=self.root,
                              env={**self.env, **env}, capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_image_build_and_failure(self):
        result = self.run_cli("image", CROSS_TEST_IMAGE_RC="17")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(self.calls(), [["docker", "build", "--platform", "linux/amd64",
                                        "-t", "test-cross:local", "-f",
                                        str(self.repo / "dev/Dockerfile.spark-cross"), str(self.repo / "dev")]])

    def test_command_preserves_arguments_and_mount_ownership(self):
        payload = "literal $(false); argument"
        result = self.run_cli("command", "tool", "path with spaces", payload)
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.calls()[-1]
        self.assertEqual(call[-4:], ["test-cross:local", "tool", "path with spaces", payload])
        self.assertEqual(call[call.index("--volume") + 1], f"{self.repo}:{self.repo}")
        self.assertEqual(call[call.index("--workdir") + 1], str(self.repo))
        self.assertEqual(call[call.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
        self.assertNotIn("-it", call)

    def test_default_build_and_option_forwarding(self):
        result = self.run_cli(CROSS_TEST_RUN_BUILD="1", DGPP_BUILD_JOBS="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[-2:], [["cmake", "--preset", "spark-cross"],
                                           ["cmake", "--build", "--preset", "spark-cross", "--parallel", "2"]])
        result = self.run_cli("build", "--target", "target with spaces", CROSS_TEST_RUN_BUILD="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[-1], ["cmake", "--build", "--preset", "spark-cross",
                                          "--parallel", "3", "--target", "target with spaces"])

    def test_configure_failure_prevents_build(self):
        result = self.run_cli("build", CROSS_TEST_RUN_BUILD="1", CROSS_TEST_CONFIGURE_RC="19")
        self.assertEqual(result.returncode, 19)
        self.assertEqual([c for c in self.calls() if c[0] == "cmake"], [["cmake", "--preset", "spark-cross"]])

    def test_build_failure_propagates(self):
        result = self.run_cli("build", CROSS_TEST_RUN_BUILD="1", CROSS_TEST_BUILD_RC="23")
        self.assertEqual(result.returncode, 23)

    def test_missing_image_explains_recovery_without_running_container(self):
        result = self.run_cli("command", "tool", CROSS_TEST_INSPECT_RC="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("scripts/spark-cross image", result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_invalid_input_rejected_before_docker(self):
        cases = [(("unknown",), {}, "Usage:"),
                 (("command",), {}, "command requires a program")]
        cases.extend((("build",), {"DGPP_BUILD_JOBS": value}, "DGPP_BUILD_JOBS")
                     for value in ("0", "-1", "abc", "1 2"))
        for args, env, message in cases:
            with self.subTest(args=args, env=env):
                result = self.run_cli(*args, **env)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
        self.assertEqual(self.calls(), [])

    def test_help_needs_no_docker(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_shell_options_and_container_exit_status(self):
        result = self.run_cli("shell", "-c", "exit 7", CROSS_TEST_RUN_RC="7")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.calls()[-1][-4:], ["test-cross:local", "bash", "-c", "exit 7"])

    def test_linked_worktree_mounts_common_git_metadata_read_only(self):
        git_env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True, env=git_env)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test",
                        "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture", "--allow-empty"], check=True, env=git_env)
        linked = self.root / "linked worktree"
        subprocess.run(["git", "-C", str(self.repo), "worktree", "add", "-q", "--detach", str(linked)], check=True, env=git_env)
        (linked / "scripts").mkdir()
        shutil.copy2(self.script, linked / "scripts/spark-cross")
        self.script = linked / "scripts/spark-cross"
        result = self.run_cli("command", "git", "rev-parse", "HEAD")
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.calls()[-1]
        mounts = [call[i + 1] for i, arg in enumerate(call) if arg == "--volume"]
        self.assertEqual(mounts, [f"{linked}:{linked}", f"{self.repo}/.git:{self.repo}/.git:ro"])


if __name__ == "__main__":
    unittest.main()
