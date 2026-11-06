"""Exercise the actual launcher without CUDA or SSH: ownership and cancellation."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile

from heretic_worker_check import Client, config_data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_process


def check(binary, metadata):
    with tempfile.TemporaryDirectory(prefix="heretic-launcher-") as tmp:
        root = Path(tmp)
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text(Path(metadata).read_text())
        config = root / "research.json"
        config.write_text(json.dumps(config_data(model)))
        (root / "site.env").write_text("")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "DGPP_ENV_FILE": str(root / "site.env"),
               "DGPP_NODES": "127.0.0.1", "DGPP_NODE_OVERRIDES": "{}", "DGPP_NO_SWAP": "0"}
        for mode in ("close", "disconnect", "cancel", "disabled", "no_swap"):
            log_dir = root / mode
            run_env = {**env, "DGPP_LOG_DIR": str(log_dir)}
            if mode == "no_swap":
                run_env["DGPP_NO_SWAP"] = "1"
            data = config_data(model)
            data["engine"]["heretic_research"] = mode != "disabled"
            config.write_text(json.dumps(data))
            parent, child = socket.socketpair()
            parent.settimeout(15)
            with tempfile.TemporaryFile() as logs:
                proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/dgpp-heretic-cluster"),
                    "--config", str(config), "--control-fd", str(child.fileno()), "--worker-binary", binary],
                    pass_fds=(child.fileno(),), env=run_env, stdout=logs, stderr=logs)
                child.close()
                try:
                    if mode in ("disabled", "no_swap"):
                        assert proc.wait(timeout=10) != 0
                        assert not list(log_dir.glob("heretic/*/r0.state.json"))
                        logs.seek(0)
                        message = logs.read().decode()
                        assert ("disabled" if mode == "disabled" else "DGPP_NO_SWAP=0") in message, message
                        continue
                    client = Client(parent)
                    assert client.call("describe")[0]["catalog"]["world"] == 1
                    states = list(log_dir.glob("heretic/*/r0.state.json"))
                    assert len(states) == 1 and cluster_process.running(states[0])
                    if mode == "close":
                        client.call("close")
                    elif mode == "disconnect":
                        parent.close()
                    else:
                        proc.send_signal(signal.SIGTERM)
                    assert proc.wait(timeout=15) == (143 if mode == "cancel" else 0)
                    assert cluster_process.running(states[0]) is None, "launcher leaked its worker"
                except BaseException:
                    logs.seek(0)
                    print(logs.read().decode(), file=sys.stderr)
                    raise
                finally:
                    parent.close()
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=15)
        print("research launcher: flag gate, inherited socket, close/disconnect/cancel and swap-policy rejection passed")


if __name__ == "__main__":
    check(*sys.argv[1:])
