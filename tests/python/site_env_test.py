import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import site_env


class SiteEnvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env_file = self.root / ".env"
        self.env_file.write_text(
            'DGPP_NODES="head peer1 peer2 peer3"\n'
            'DGPP_SSH_USER="ops"\n'
            'DGPP_HTTP_PORT=18888\n'
            'HF_ACCESS_TOKEN=credential-must-not-be-loaded\n'
        )
        self.config = self.root / "deployment.json"
        self.config.write_text(json.dumps({
            "model": "org/model", "world_size": 2,
            "engine": {"mtp": True}, "paths": {"resident_cache": "/cache/model"},
        }))
        self.environ = {
            "DGPP_ENV_FILE": str(self.env_file),
            "DGPP_CLUSTER_CONFIG": str(self.config),
            "PATH": os.environ.get("PATH", os.defpath),
        }

    def values(self, **overrides):
        return site_env.settings({**self.environ, **overrides})

    def shell(self, command, **overrides):
        return subprocess.run(
            ["bash", "-c", '. "$1" || exit 1; ' + command, "bash", str(ROOT / "scripts/cluster_env.sh")],
            env={**self.environ, **overrides}, text=True, capture_output=True,
        )

    def test_quoted_values_comments_and_allowlist(self):
        self.env_file.write_text(
            '# site\nexport DGPP_NODES = "head peer" # rank order\n'
            "DGPP_SSH_USER = 'ops'\nDGPP_HTTP_PORT=18888 # HTTP\n"
            'HF_ACCESS_TOKEN="unterminated-unrelated-credential\n'
            'HOME=/do/not/override\n'
        )
        self.assertEqual(site_env.read_env(self.env_file), {
            "DGPP_NODES": "head peer", "DGPP_SSH_USER": "ops", "DGPP_HTTP_PORT": "18888",
        })

    def test_exported_values_override_dotenv(self):
        values = self.values(DGPP_NODES="override", DGPP_SSH_USER="other")
        self.assertEqual(site_env.site_nodes(values), ["override"])
        self.assertEqual(site_env.ssh_user(values), "other")

    def test_repo_env_is_found_without_explicit_path(self):
        with patch.object(site_env, "ROOT", self.root):
            values = site_env.settings({})
            self.assertEqual(site_env.site_nodes(values), ["head", "peer1", "peer2", "peer3"])
            self.assertEqual(site_env.config_path(values), str(self.root / "deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json"))

    def test_relative_default_deployment_is_based_at_repo_root(self):
        with patch.object(site_env, "ROOT", self.root):
            self.assertEqual(site_env.config_path({"DGPP_CLUSTER_CONFIG": "deploy/model.json"}),
                             str(self.root / "deploy/model.json"))

    def test_explicit_config_overrides_environment_default(self):
        other = self.root / "other.json"
        other.write_text(json.dumps({"model": "org/other", "world_size": 1}))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/dgpp-cluster"), "resolve", "--config", str(other)],
            env=self.environ, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["model"], "org/other")

    def test_relative_env_path_survives_shell_directory_change(self):
        result = subprocess.run(
            ["bash", "-c", '. "$1" || exit 1; cd /; dgpp_head', "bash", str(ROOT / "scripts/cluster_env.sh")],
            cwd=self.root, env={**self.environ, "DGPP_ENV_FILE": ".env"}, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "head\n")

    def test_literal_values_never_execute_shell(self):
        marker = self.root / "executed"
        literal = f"$(touch {marker})"
        self.env_file.write_text(self.env_file.read_text() + f"DGPP_LOG_DIR='{literal}'\n")
        result = self.shell('printf "%s" "$DGPP_LOG_DIR"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, literal)
        self.assertFalse(marker.exists())

    def test_shell_and_python_agree_without_exporting_credentials(self):
        result = self.shell('dgpp_head; dgpp_nodes; dgpp_peers; dgpp_ssh_user; dgpp_http_port; printf "%s" "${HF_ACCESS_TOKEN-unset}"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["head", "head peer1", "peer1", "ops", "18888", "unset"])

    def test_one_two_and_four_nodes_preserve_rank_order(self):
        for world in (1, 2, 4):
            with self.subTest(world=world):
                cfg = json.loads(self.config.read_text())
                cfg["world_size"] = world
                self.config.write_text(json.dumps(cfg))
                resolved = site_env.resolve_config(self.config, self.values())
                self.assertEqual(resolved["nodes"], ["head", "peer1", "peer2", "peer3"][:world])
                result = self.shell("dgpp_nodes; dgpp_peers")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, " ".join(resolved["nodes"]) + "\n" + " ".join(resolved["nodes"][1:]) + "\n")

    def test_resolved_config_keeps_model_settings_and_excludes_env(self):
        original = self.config.read_text()
        resolved = site_env.resolve_config(self.config, self.values())
        self.assertEqual(resolved["model"], "org/model")
        self.assertEqual(resolved["engine"], {"mtp": True})
        self.assertEqual(resolved["paths"]["resident_cache"], "/cache/model")
        self.assertEqual(resolved["ports"]["http"], 18888)
        self.assertEqual(resolved["ssh_user"], "ops")
        self.assertNotIn("world_size", resolved)
        self.assertNotIn("credential", json.dumps(resolved))
        self.assertEqual(self.config.read_text(), original)

    def test_http_body_limit_survives_deployment_resolution(self):
        cfg = json.loads(self.config.read_text())
        cfg["http"] = {"max_body_bytes": 5 * 1024**3}
        self.config.write_text(json.dumps(cfg))
        resolved = site_env.resolve_config(self.config, self.values())
        self.assertEqual(resolved["http"]["max_body_bytes"], 5 * 1024**3)
        self.assertEqual(resolved["http"]["port"], 18888)
        self.assertEqual(resolved["http"]["bind_host"], "127.0.0.1")

    def test_invalid_http_body_limits_fail(self):
        cfg = json.loads(self.config.read_text())
        for value in (0, -1, True, "256MiB", 1.5, None, 1 << 63):
            with self.subTest(value=value):
                cfg["http"] = {"max_body_bytes": value}
                self.config.write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError, "http.max_body_bytes"):
                    site_env.resolve_config(self.config, self.values())

    def test_sse_ping_interval_survives_deployment_resolution(self):
        cfg = json.loads(self.config.read_text())
        for interval in (-1, 1, 30, 2147483647):
            with self.subTest(interval=interval):
                cfg["http"] = {"sse_ping_interval": interval}
                self.config.write_text(json.dumps(cfg))
                resolved = site_env.resolve_config(self.config, self.values())
                self.assertEqual(resolved["http"]["sse_ping_interval"], interval)
                self.assertEqual(resolved["http"]["port"], 18888)

    def test_invalid_sse_ping_intervals_fail(self):
        cfg = json.loads(self.config.read_text())
        for interval in (0, -2, True, None, "30", 1.5, 1.0, 2147483648, 1e100):
            with self.subTest(interval=interval):
                cfg["http"] = {"sse_ping_interval": interval}
                self.config.write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError, "http.sse_ping_interval"):
                    site_env.resolve_config(self.config, self.values())

    def test_empty_user_has_consistent_login_fallback(self):
        with patch.object(site_env.getpass, "getuser", return_value="fallback"):
            self.assertEqual(site_env.ssh_user(self.values(DGPP_SSH_USER="")), "fallback")

    def test_missing_nodes_and_insufficient_nodes_fail(self):
        for value in ("", "head"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                site_env.resolve_config(self.config, self.values(DGPP_NODES=value))

    def test_invalid_values_fail(self):
        cases = [
            {"DGPP_NODES": "head head"}, {"DGPP_NODES": "head;command peer"},
            {"DGPP_SSH_USER": "-oProxyCommand=bad"}, {"DGPP_HTTP_PORT": "65536"},
            {"DGPP_HTTP_PORT": "1.5"}, {"DGPP_FABRIC_PORT": "29971"},
            {"DGPP_STAGE_DIR": "/"}, {"DGPP_STAGE_DIR": "/tmp/../"},
            {"DGPP_RELEASE_DIR": "/tmp/x;command"},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                site_env.resolve_config(self.config, self.values(**case))

    def test_metrics_port_is_optional_and_validated(self):
        self.assertNotIn("metrics", site_env.resolve_config(self.config, self.values())["ports"])
        resolved = site_env.resolve_config(self.config, self.values(DGPP_METRICS_PORT="29972"))
        self.assertEqual(resolved["ports"]["metrics"], 29972)
        for value, message in (("70000", "DGPP_METRICS_PORT must be 0"), ("-1", "DGPP_METRICS_PORT must be 0"),
                               ("29970", "must differ from DGPP_FABRIC_PORT")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                site_env.resolve_config(self.config, self.values(DGPP_METRICS_PORT=value))

    def test_missing_explicit_env_file_fails(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            site_env.settings({"DGPP_ENV_FILE": str(self.root / "missing")})

    def test_bad_quote_and_duplicate_setting_fail_without_value_in_error(self):
        for content in ('DGPP_SSH_USER="private-value\n', 'DGPP_SSH_USER=a\nDGPP_SSH_USER=private-value\n'):
            self.env_file.write_text(content)
            with self.assertRaises(ValueError) as error:
                site_env.read_env(self.env_file)
            self.assertNotIn("private-value", str(error.exception))

    def test_legacy_json_and_invalid_world_sizes_fail(self):
        for extra in ({"nodes": ["old"]}, {"ssh_user": "old"}, {"ports": {}},
                      {"world_size": True}, {"world_size": 0}, {"world_size": -1},
                      {"world_size": "2"}, {"paths": {"stage_dir": "/old"}}):
            cfg = {"model": "org/model", "world_size": 2, **extra}
            self.config.write_text(json.dumps(cfg))
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                site_env.resolve_config(self.config, self.values())

    def test_any_rank_count_the_site_has_nodes_for_resolves(self):
        # No allow-list of world sizes: a rank count the engine may or may not
        # support still resolves here, and fails later where the reason is known
        # (geometry that does not divide, or a memory plan that does not fit).
        cfg = json.loads(self.config.read_text())
        cfg["world_size"] = 3
        self.config.write_text(json.dumps(cfg))
        resolved = site_env.resolve_config(self.config, self.values())
        self.assertEqual(resolved["nodes"], ["head", "peer1", "peer2"])
        # A world the site cannot staff is still refused, by the node list.
        cfg["world_size"] = 5
        self.config.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError) as error:
            site_env.resolve_config(self.config, self.values())
        self.assertIn("only 4", str(error.exception))

    def test_all_deployment_examples_resolve_without_site_fields(self):
        for path in (ROOT / "deploy").glob("*.example.json"):
            with self.subTest(path=path.name):
                cfg = json.loads(path.read_text())
                self.assertFalse(set(cfg) & {"nodes", "ssh_user", "ports"})
                resolved = site_env.resolve_config(path, self.values())
                self.assertEqual(len(resolved["nodes"]), cfg["world_size"])
                self.assertEqual(resolved["engine"], cfg["engine"])

    def test_resolve_command_is_read_only_and_works_outside_repo(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/dgpp-cluster"), "resolve", "--config", str(self.config)],
            env=self.environ, cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), site_env.resolve_config(self.config, self.values()))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [".env", "deployment.json"])

    def test_cpp_fixture_matches_resolved_example(self):
        values = {**site_env.DEFAULTS, **site_env.read_env(ROOT / ".env.example"), "DGPP_SSH_USER": "ops"}
        resolved = site_env.resolve_config(ROOT / "deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json", values)
        self.assertEqual(resolved, json.loads((ROOT / "tests/fixtures/cluster.resolved.json").read_text()))

    def test_packaged_layout_finds_its_own_site_settings(self):
        release = self.root / "release"
        (release / "scripts").mkdir(parents=True)
        (release / "deploy").mkdir()
        for name in ("site_env.py", "cluster_process.py", "dgpp-cluster"):
            (release / "scripts" / name).write_text((ROOT / "scripts" / name).read_text())
        (release / ".env").write_text(self.env_file.read_text())
        (release / "deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json").write_text(self.config.read_text())
        result = subprocess.run(
            [sys.executable, str(release / "scripts/dgpp-cluster"), "resolve"],
            cwd="/", env={"PATH": self.environ["PATH"]}, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["nodes"], ["head", "peer1"])

    def test_launcher_stages_resolved_config_and_boots_head_with_it(self):
        module = runpy.run_path(str(ROOT / "scripts/dgpp-cluster"))
        args = argparse.Namespace(config=str(self.config), log_dir=str(self.root / "logs"), knobs="")
        with patch.dict(os.environ, self.environ, clear=True):
            cluster = module["Cluster"](args)
            with patch.object(cluster, "ssh", return_value=True), patch.object(cluster, "scp_to", return_value=True) as scp:
                self.assertTrue(cluster.stage())
            staged = [call.args[1] for call in scp.call_args_list]
            self.assertIn(cluster.config_path, staged)
            self.assertNotIn(str(self.env_file), staged)
            self.assertNotIn(str(self.config), staged)
            self.assertEqual(json.loads(Path(cluster.config_path).read_text()), cluster.cfg)
            with patch.object(module["cluster_process"], "launch") as popen:
                popen.return_value.pid = 12345
                cluster.boot_head()
                command = popen.call_args.args[1]
                self.assertEqual(command[command.index("--config") + 1], cluster.config_path)
            with patch.object(module["subprocess"], "run") as remote:
                remote.return_value.returncode = 0
                cluster.spawn_peer(1, "peer1")
                command = remote.call_args.args[0][-1]
                self.assertIn(f"--config {cluster.stage_dir}/cluster.json --rank 1", command)
                self.assertNotIn("HF_ACCESS_TOKEN", command)
                self.assertNotIn("DGPP_ENV_FILE", command)

    def test_diagnostic_settings_reach_head_and_peer_with_precedence(self):
        self.env_file.write_text(self.env_file.read_text() +
                                 'DGPP_LOG_LEVEL=debug\nDGPP_MLOCK=off\n')
        module = runpy.run_path(str(ROOT / "scripts/dgpp-cluster"))
        args = argparse.Namespace(config=str(self.config), log_dir=str(self.root / "logs"), knobs="")
        cases = [
            ({}, [("debug", "off"), ("debug", "off")]),
            ({"DGPP_LOG_LEVEL": "info", "DGPP_MLOCK": "on"},
             [("info", "on"), ("info", "on")]),
            ({"DGPP_LOG_LEVEL": "info", "DGPP_MLOCK": "on",
              "DGPP_NODE_OVERRIDES": json.dumps({
                  "head": {"DGPP_LOG_LEVEL": "trace", "DGPP_MLOCK": "off"},
                  "peer1": {"DGPP_LOG_LEVEL": "debug", "DGPP_MLOCK": "off"},
              })}, [("trace", "off"), ("debug", "off")]),
        ]
        for exported, expected in cases:
            with self.subTest(exported=exported), patch.dict(
                    os.environ, {**self.environ, **exported}, clear=True):
                cluster = module["Cluster"](args)
                Path(cluster.log_dir).mkdir(parents=True, exist_ok=True)
                for rank, values in enumerate(expected):
                    self.assertEqual(cluster.cfg["node_env"][rank]["DGPP_LOG_LEVEL"], values[0])
                    self.assertEqual(cluster.cfg["node_env"][rank]["DGPP_MLOCK"], values[1])
                with patch.object(module["cluster_process"], "launch") as launch:
                    launch.return_value.pid = 12345
                    cluster.boot_head()
                    env = launch.call_args.args[4]
                    self.assertEqual((env["DGPP_LOG_LEVEL"], env["DGPP_MLOCK"]), expected[0])
                with patch.object(module["subprocess"], "run") as remote:
                    remote.return_value.returncode = 0
                    cluster.spawn_peer(1, "peer1")
                    words = shlex.split(remote.call_args.args[0][-1])
                    # Shell assignments are applied left to right; a node
                    # override must win over the launcher's default log level.
                    assignments = dict(word.split("=", 1) for word in words[:words.index("python3")]
                                       if word.startswith("DGPP_") and "=" in word)
                    self.assertEqual((assignments["DGPP_LOG_LEVEL"], assignments["DGPP_MLOCK"]),
                                     expected[1])

    def test_operational_wrappers_stop_on_invalid_env_before_running(self):
        scripts = [p for p in (ROOT / "scripts").glob("*.sh")
                   if p.name != "cluster_env.sh" and 'cluster_env.sh" || exit 1' in p.read_text()]
        self.assertGreater(len(scripts), 20)
        for path in scripts:
            with self.subTest(script=path.name):
                result = subprocess.run(["bash", str(path)], env={**self.environ, "DGPP_ENV_FILE": str(self.root / "missing")},
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("site environment file does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
