"""Portable setup and process ownership checks; no GPU, SSH or model needed."""
import argparse
import json
import os
import re
from pathlib import Path
import runpy
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_doctor
import cluster_process
import prepare_data
import site_env


class PortabilityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "cluster.json"
        self.config.write_text(json.dumps({"model": "org/model", "world_size": 1}))
        self.env_file = self.root / ".env"
        self.env_file.write_text('DGPP_NODES="127.0.0.1"\n')
        self.env = {"PATH": os.environ["PATH"], "DGPP_ENV_FILE": str(self.env_file),
                    "DGPP_CLUSTER_CONFIG": str(self.config)}
        self.values = site_env.settings(self.env)

    def test_localhost_and_deployment_http_override(self):
        resolved = site_env.resolve_config(self.config, self.values)
        self.assertEqual(resolved["http"], {"bind_host": "127.0.0.1", "port": 18080})
        self.config.write_text(json.dumps({"model": "org/model", "world_size": 1,
                                         "http": {"bind_host": "0.0.0.0", "port": 8080}}))
        resolved = site_env.resolve_config(self.config, self.values)
        self.assertEqual(resolved["ports"]["http"], 8080)
        with patch.dict(os.environ, self.env, clear=True):
            self.assertEqual(site_env.default_host(), "127.0.0.1")
            self.assertEqual(site_env.http_port(), 8080)

    def test_nvme_cache_block_passes_through_and_is_checked(self):
        # The cold tier's top-level block (issue #26) rides the resolved config
        # as written; its keys and types are checked by name.
        self.config.write_text(json.dumps({"model": "org/model", "world_size": 1,
                                         "nvme_cache": {"enabled": True, "path": "/nvme/dgpp",
                                                        "capacity_gib": 64, "min_tokens": 2048}}))
        resolved = site_env.resolve_config(self.config, self.values)
        self.assertEqual(resolved["nvme_cache"], {"enabled": True, "path": "/nvme/dgpp",
                                                  "capacity_gib": 64, "min_tokens": 2048})
        self.config.write_text(json.dumps({"model": "org/model", "world_size": 1}))
        self.assertNotIn("nvme_cache", site_env.resolve_config(self.config, self.values))
        for nvme in ({"enabled": True}, {"enabled": 1}, {"capacity_gib": -1}, {"capacity_gib": True},
                     {"path": ""}, {"min_tokens": 1.5}, {"size": 3}, []):
            with self.subTest(nvme=nvme):
                self.config.write_text(json.dumps({"model": "org/model", "world_size": 1, "nvme_cache": nvme}))
                with self.assertRaises(ValueError):
                    site_env.resolve_config(self.config, self.values)

    def test_http_types_and_ports_rejected(self):
        for http in ({"bind_host": 0}, {"bind_host": "localhost"}, {"port": True}, {"port": 0}, {"host": "a"}):
            with self.subTest(http=http):
                self.config.write_text(json.dumps({"model": "org/model", "world_size": 1, "http": http}))
                with self.assertRaises(ValueError):
                    site_env.resolve_config(self.config, self.values)

    def test_cache_and_lane_overrides(self):
        values = {**self.values, "DGPP_NODES": "head peer", "HF_HOME": "~/cache",
                  "DGPP_ROCE_DEVICES": "nic0 nic1", "DGPP_ROCE_GID_INDICES": "3 4",
                  "DGPP_NODE_OVERRIDES": json.dumps({"peer": {"HF_HUB_CACHE": "/cache/hub", "DGPP_ROCE_DEVICES": "nic2 nic3"}})}
        envs = site_env.node_environments(values)
        self.assertEqual(envs[0]["HF_HUB_CACHE"], "~/cache/hub")
        self.assertEqual(envs[1]["HF_HUB_CACHE"], "/cache/hub")
        self.assertEqual(envs[1]["DGPP_ROCE_DEVICES"], "nic2 nic3")
        self.assertIn('"$HOME"/', site_env.shell_prefix(envs[0]))
        for override in ({"outsider": {}}, {"head": {"HF_TOKEN": "secret"}}, {"head": {"DGPP_ROCE_GID_INDICES": "1"}}):
            with self.assertRaises(ValueError):
                site_env.node_environments({**values, "DGPP_NODE_OVERRIDES": json.dumps(override)})

    def test_paths_are_namespaced(self):
        a = site_env.run_paths(self.config, self.values)
        b = site_env.run_paths(self.root / "other.json", self.values)
        self.assertNotEqual(a["stage_dir"], b["stage_dir"])
        self.assertNotEqual(a["log_dir"], b["log_dir"])
        self.assertEqual(site_env.run_paths(self.config, self.values, "/logs")["log_dir"], "/logs")

    def test_process_record_owns_only_launched_session(self):
        state, log = self.root / "state.json", self.root / "run.log"
        proc = cluster_process.launch(state, ["/bin/sleep", "30"], log, self.root)
        try:
            self.assertEqual(cluster_process.running(state)["pid"], proc.pid)
            with self.assertRaises(ValueError):
                cluster_process.launch(state, ["/bin/sleep", "30"], log, self.root)
            self.assertTrue(cluster_process.send_signal(state, signal.SIGTERM))
            proc.wait(timeout=5)
            self.assertIsNone(cluster_process.running(state))
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_stale_identity_never_signals_reused_pid(self):
        state = self.root / "state.json"
        record = cluster_process.identity(os.getpid())
        record["start"] = "different-start"
        state.write_text(json.dumps(record))
        with patch.object(os, "killpg") as kill:
            self.assertFalse(cluster_process.send_signal(state, signal.SIGKILL))
            kill.assert_not_called()

    def test_unrecorded_status_does_not_create_files(self):
        self.assertIsNone(cluster_process.running(self.root / "missing/state.json"))
        self.assertFalse((self.root / "missing").exists())

    def checkpoint(self):
        snapshot = self.root / "models--org--model/snapshots/revision"
        snapshot.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json"):
            (snapshot / name).write_text("{}")
        (snapshot / "chat_template.jinja").write_text("{{ messages }}")
        header = json.dumps({"weight": {"data_offsets": [0, 4], "dtype": "F32", "shape": [1]}}).encode()
        (snapshot / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))
        return snapshot

    def test_complete_checkpoint_and_truncation(self):
        snapshot = self.checkpoint()
        self.assertEqual(cluster_doctor.cached_snapshot("org/model", self.root), snapshot)
        self.assertGreater(cluster_doctor.checkpoint_size(snapshot), 4)
        shard = snapshot / "model.safetensors"
        shard.write_bytes(shard.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "length"):
            cluster_doctor.checkpoint_size(snapshot)

    def test_snapshot_ambiguity_and_missing_shard(self):
        snapshot = self.checkpoint()
        (snapshot.parent / "other").mkdir()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            cluster_doctor.cached_snapshot("org/model", self.root)
        ref = snapshot.parent.parent / "refs/main"
        ref.parent.mkdir()
        ref.write_text("revision")
        self.assertEqual(cluster_doctor.cached_snapshot("org/model", self.root), snapshot)
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
        with self.assertRaises(OSError):
            cluster_doctor.checkpoint_size(snapshot)

    def test_data_preparation_preserves_existing_content(self):
        path = self.root / "dataset"
        prepare_data.write_new(path, b"original")
        prepare_data.write_new(path, b"original")
        with self.assertRaises(ValueError):
            prepare_data.write_new(path, b"different")
        self.assertEqual(path.read_bytes(), b"original")

    def test_unpopulated_sysfs_gid_slots_are_skipped(self):
        port = self.root / "port"
        (port / "gid_attrs/types").mkdir(parents=True)
        (port / "gids").mkdir()
        (port / "gid_attrs/types/0").touch()
        (port / "gid_attrs/types/1").write_text("RoCE v2")
        (port / "gids/1").write_text("0000:0000:0000:0000:0000:ffff:c000:0201")
        read = Path.read_text
        def read_slot(path, *args, **kwargs):
            if path.name == "0":
                raise OSError(22, "Invalid argument")
            return read(path, *args, **kwargs)
        with patch.object(Path, "read_text", read_slot):
            self.assertEqual(cluster_doctor.roce_gids(port), [("1", "0000:0000:0000:0000:0000:ffff:c000:0201")])

    def test_humaneval_acknowledgement_before_network_or_output(self):
        out = self.root / "results"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/serve_eval.py"), "invalid", "1", "--out", str(out)],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--allow-code-execution", result.stderr)
        self.assertFalse(out.exists())

    def test_peer_procedures_reject_single_node_before_launch(self):
        for name in ("serve_stop_check.sh", "serve_failure_drill.sh", "serve_soak_run.sh"):
            result = subprocess.run(["bash", str(ROOT / "scripts" / name)], env=self.env,
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires a world with peers", result.stderr)

    def test_scripts_index_and_setup_document_links(self):
        index = (ROOT / "scripts/README.md").read_text()
        for path in (ROOT / "scripts").iterdir():
            if path.is_file():
                self.assertIn(f"[{path.name}]({path.name})", index)
        for name in ("README.md", "CONTRIBUTING.md", "deploy/README.md", "scripts/README.md", "docs/getting-started.md", "docs/dependencies.md", "docs/networking.md", "docs/testing.md", "docs/operations.md"):
            source = ROOT / name
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", source.read_text()):
                if target.startswith(("http:", "https:", "#")):
                    continue
                filename, _, anchor = target.partition("#")
                destination = source.parent / filename
                self.assertTrue(destination.exists(), f"broken link in {name}: {target}")
                if anchor and destination.suffix == ".md":
                    headings = re.findall(r"^#{1,6} (.+)$", destination.read_text(), re.MULTILINE)
                    anchors = {re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-") for heading in headings}
                    self.assertIn(anchor, anchors, f"broken heading link in {name}: {target}")

    def test_contributor_cmake_minimum_matches_build(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        minimum = re.search(r"cmake_minimum_required\(VERSION ([0-9.]+)\)", cmake).group(1)
        self.assertIn(f"CMake {minimum}+", (ROOT / "CONTRIBUTING.md").read_text())

    def test_deployment_filenames_describe_their_settings(self):
        models = {
            "HawkBearPig/GLM-5.3-Flash-NVFP4-FP8": "glm-5.3-flash_nvfp4-fp8",
            "HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64": "glm-5.3_int4-int8",
            "nvidia/GLM-4.7-NVFP4": "glm-4.7_nvfp4",
            "Qwen/Qwen3.8-Flash-Next-FP8": "qwen-3.8-flash-next_fp8",
            "nvidia/Qwen3.8-Flash-Next-NVFP4": "qwen-3.8-flash-next_nvfp4",
            "RadixArk/Qwen3.8-Flash-Next-NVFP4": "qwen-3.8-flash-next_nvfp4-radixark",
            "deepseek-ai/DeepSeek-V4.1-Flash": "deepseek-v4.1-flash_mxfp4-fp8",
            "XiaomiMiMo/MiMo-V2.6-Flash-RL": "mimo-v2.6-flash_mxfp4-fp8",
        }
        values = {**site_env.DEFAULTS, "DGPP_NODES": "head peer1 peer2 peer3", "DGPP_SSH_USER": "ops"}
        templates = list((ROOT / "deploy").glob("*.example.json"))
        self.assertTrue(templates)
        index = (ROOT / "deploy/README.md").read_text()
        # One template per model, quant and world (2026-09-14): every template
        # enables MTP with the decode graph; the shapes a template does not
        # name are boot knobs, listed in the catalogue.
        seen = set()
        for path in templates:
            with self.subTest(path=path.name):
                cfg = json.loads(path.read_text())
                engine = cfg["engine"]
                self.assertTrue(engine["mtp"], "every template enables MTP (the plain world is --no-mtp)")
                self.assertTrue(engine["decode_graph"])
                # The name is the shape: cluster_<model>_<quant>_w<n>, with an
                # optional trailing _<variant> for a template that deviates from
                # that shape in one documented engine setting (today the Qwen
                # 512K YaRN ramp) instead of in its size. A base shape stays
                # unique, and a variant may repeat one only with its suffix;
                # deploy/README.md carries the rule and the catalogue.
                shape = re.fullmatch(
                    rf"cluster_{re.escape(models[cfg['model']])}_w([0-9]+)(_[a-z0-9]+)?",
                    path.name.removesuffix(".example.json"))
                self.assertIsNotNone(
                    shape, f"a template is named cluster_<model>_<quant>_w<n>[_variant]: {path.name}")
                self.assertEqual(int(shape.group(1)), cfg["world_size"],
                                 "the world size in the name is the JSON's")
                stem = f"cluster_{models[cfg['model']]}_w{cfg['world_size']}{shape.group(2) or ''}"
                self.assertNotIn(stem, seen, "one template per model, quant, world and variant")
                seen.add(stem)
                self.assertEqual(path.name, stem + ".example.json")
                self.assertIn(f"]({path.name})", index)
                resolved = site_env.resolve_config(path, values)
                self.assertEqual(len(resolved["nodes"]), cfg["world_size"])
                self.assertEqual(resolved["engine"], engine)

    def cluster(self):
        module = runpy.run_path(str(ROOT / "scripts/dgpp-cluster"))
        with patch.dict(os.environ, self.env, clear=True):
            return module["Cluster"](argparse.Namespace(config=str(self.config), log_dir=str(self.root / "logs"), knobs=""))

    def test_failed_preflight_does_not_stage_or_signal(self):
        cluster = self.cluster()
        with patch.object(cluster_doctor, "check_cluster", return_value=1), patch.object(cluster, "stage") as stage, patch.object(cluster, "signal_rank") as stop:
            self.assertEqual(cluster.up(), 1)
            stage.assert_not_called()
            stop.assert_not_called()

    def test_existing_deployment_refused_before_staging(self):
        cluster = self.cluster()
        with patch.object(cluster, "head_pid", return_value=123), patch.object(cluster, "stage") as stage:
            with self.assertRaisesRegex(ValueError, "already running"):
                cluster.up()
            stage.assert_not_called()

    def test_start_failure_cleans_recorded_head(self):
        cluster = self.cluster()
        cluster.skip_preflight = True
        with patch.object(cluster, "stage", return_value=True), patch.object(cluster, "start_ranks", side_effect=RuntimeError("failed")), patch.object(cluster, "signal_rank") as stop:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                cluster.up()
            stop.assert_called_once_with(0, "KILL")


if __name__ == "__main__":
    unittest.main()
