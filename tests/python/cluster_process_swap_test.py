"""Check swap policy validation/propagation and real, small systemd user scopes."""
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_process
import cluster_doctor


class ProcessSwapTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.state = self.directory / "process.json"
        self.log = self.directory / "process.log"
        self.env = {**os.environ, "DGPP_NO_SWAP": "1"}

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)

    def require_systemd_run(self):
        if not shutil.which("systemd-run"):
            self.skipTest("systemd-run is unavailable")

    def require_swap_scope(self):
        self.require_systemd_run()
        if not shutil.which("systemctl"):
            self.skipTest("systemctl is unavailable")
        try:
            manager = subprocess.run(
                ["systemctl", "--user", "show", "--property=ControlGroup", "--value"],
                capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            self.skipTest("systemd user manager did not respond")
        if manager.returncode:
            self.skipTest("systemd user manager is unavailable")
        group = manager.stdout.strip()
        if not group.startswith("/"):
            self.skipTest("systemd user manager has no cgroup")
        cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
        try:
            controllers = (cgroup / "cgroup.controllers").read_text().split()
            (cgroup / "memory.swap.max").read_text()
            (cgroup / "memory.swap.current").read_text()
        except (FileNotFoundError, PermissionError):
            self.skipTest("cgroup v2 memory/swap controls are unavailable")
        if "memory" not in controllers:
            self.skipTest("cgroup v2 memory controller is unavailable for user scopes")

    def test_scope_has_zero_swap_and_recorded_group_can_stop_it(self):
        self.require_swap_scope()
        ready = self.directory / "ready.json"
        code = """
import json, os, signal, sys
from pathlib import Path
group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
             if line.startswith('0::'))
cg = Path('/sys/fs/cgroup') / group.lstrip('/')
memory = bytearray(8 * 1024 * 1024)
Path(sys.argv[1]).write_text(json.dumps({
    'pid': os.getpid(), 'sid': os.getsid(0), 'group': group,
    'swap_max': (cg / 'memory.swap.max').read_text().strip(),
    'swap_current': (cg / 'memory.swap.current').read_text().strip(),
    'argument': sys.argv[2], 'cwd': os.getcwd(), 'policy_env': os.environ['DGPP_NO_SWAP']}))
signal.pause()
"""
        literal = "argument with spaces, 'quotes', $ and `backticks`"
        proc = cluster_process.launch(self.state, [sys.executable, "-c", code, str(ready), literal],
                                      self.log, self.directory, self.env)
        self.addCleanup(self.stop, proc)
        deadline = time.monotonic() + 10
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), self.log.read_text())
        result = json.loads(ready.read_text())
        self.assertEqual(result['swap_max'], '0')
        self.assertEqual(result['swap_current'], '0')
        self.assertEqual(result['sid'], proc.pid)
        self.assertEqual(result['argument'], literal)
        self.assertEqual(result['cwd'], str(self.directory))
        self.assertEqual(result['policy_env'], '1')
        self.assertIsNotNone(cluster_process.running(self.state))
        self.assertTrue(cluster_process.send_signal(self.state, signal.SIGTERM))
        proc.wait(timeout=10)
        self.assertIsNone(cluster_process.running(self.state))

    def test_scope_failure_does_not_run_the_command_without_protection(self):
        # A working manager is deliberately unnecessary for this failure path.
        self.require_systemd_run()
        marker = self.directory / "must-not-exist"
        env = {**self.env, "XDG_RUNTIME_DIR": str(self.directory),
               "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(self.directory / 'missing-bus')}
        command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                   str(marker)]
        try:
            proc = cluster_process.launch(self.state, command, self.log, self.directory, env)
        except RuntimeError:
            pass  # The failed scope may exit before its identity is recorded.
        else:
            self.addCleanup(self.stop, proc)
            self.assertNotEqual(proc.wait(timeout=10), 0)
        self.assertFalse(marker.exists())

    def test_invalid_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'DGPP_NO_SWAP must be 0 or 1'):
            cluster_process.launch(self.state, ['false'], self.log, self.directory,
                                   {**self.env, 'DGPP_NO_SWAP': 'yes'})
        self.assertFalse(self.state.exists())

    def version_binary(self):
        binary = self.directory / 'version probe with spaces'
        binary.write_text(f'#!{sys.executable}\n' + """
import json, sys
from pathlib import Path
group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
             if line.startswith('0::'))
cg = Path('/sys/fs/cgroup') / group.lstrip('/')
print(json.dumps({'arguments': sys.argv[1:],
                  'swap_max': (cg / 'memory.swap.max').read_text().strip(),
                  'swap_current': (cg / 'memory.swap.current').read_text().strip()}))
""")
        binary.chmod(0o700)
        return binary

    def test_version_probe_uses_zero_swap_scope(self):
        self.require_swap_scope()
        result = cluster_doctor.server_version(str(self.version_binary()), self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {'arguments': ['--version'], 'swap_max': '0', 'swap_current': '0'})

    def test_version_probe_does_not_fall_back_when_scope_fails(self):
        self.require_systemd_run()
        env = {**self.env, 'XDG_RUNTIME_DIR': str(self.directory),
               'DBUS_SESSION_BUS_ADDRESS': 'unix:path=' + str(self.directory / 'missing-bus')}
        result = cluster_doctor.server_version(str(self.version_binary()), env)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.stdout)

    def test_version_probe_rejects_invalid_policy(self):
        with self.assertRaisesRegex(ValueError, 'DGPP_NO_SWAP must be 0 or 1'):
            cluster_doctor.server_version('must-not-run', {**self.env, 'DGPP_NO_SWAP': 'yes'})

    def test_preflight_forwards_swap_policy_to_peer_version_probes(self):
        cfg = {'nodes': ['127.0.0.1', '192.0.2.2'], 'node_env': [{}, {}],
               'model': 'test/model', 'ports': {}, 'http': {'bind_host': '127.0.0.1'}, 'paths': {}}
        report = {'rank': 0, 'checks': [], 'devices': []}
        remote = subprocess.CompletedProcess([], 0, json.dumps({**report, 'rank': 1}), '')
        with mock.patch.dict(os.environ, {'DGPP_NO_SWAP': '1'}), \
                mock.patch.object(cluster_doctor, 'probe', return_value=report) as local, \
                mock.patch.object(cluster_doctor.subprocess, 'run', return_value=remote) as ssh:
            self.assertEqual(cluster_doctor.check_cluster(cfg, 'binary', '/logs', '/stage', 'user',
                                                         peer_binary='peer-binary'), 0)
        self.assertEqual(local.call_args.args[0]['env']['DGPP_NO_SWAP'], '1')
        peer_spec = json.loads(shlex.split(ssh.call_args.args[0][-1])[-1])
        self.assertEqual(peer_spec['env']['DGPP_NO_SWAP'], '1')
        self.assertEqual(cfg['node_env'], [{}, {}])

    def test_preflight_respects_resolved_node_swap_policy(self):
        cfg = {'nodes': ['127.0.0.1', '192.0.2.2'],
               'node_env': [{'DGPP_NO_SWAP': '1'}, {'DGPP_NO_SWAP': '1'}],
               'model': 'test/model', 'ports': {}, 'http': {'bind_host': '127.0.0.1'}, 'paths': {}}
        report = {'rank': 0, 'checks': [], 'devices': []}
        remote = subprocess.CompletedProcess([], 0, json.dumps({**report, 'rank': 1}), '')
        with mock.patch.dict(os.environ, {'DGPP_NO_SWAP': '0'}), \
                mock.patch.object(cluster_doctor, 'probe', return_value=report) as local, \
                mock.patch.object(cluster_doctor.subprocess, 'run', return_value=remote) as ssh:
            self.assertEqual(cluster_doctor.check_cluster(cfg, 'binary', '/logs', '/stage', 'user',
                                                         peer_binary='peer-binary'), 0)
        self.assertEqual(local.call_args.args[0]['env']['DGPP_NO_SWAP'], '1')
        peer_spec = json.loads(shlex.split(ssh.call_args.args[0][-1])[-1])
        self.assertEqual(peer_spec['env']['DGPP_NO_SWAP'], '1')


if __name__ == '__main__':
    unittest.main()
