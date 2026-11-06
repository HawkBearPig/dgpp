"""Worker worlds over the real journal, with metadata-only and native GPU gates."""
import contextlib
import json
import math
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile

from heretic_worker_check import Client, config_data, check_weight_reads, check_runtime
from glm_dsa_reference_dump import load_linear, read_safetensors_index


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def world(binary, model, count, gpu, mismatch=False):
    with tempfile.TemporaryDirectory(prefix="heretic-world-") as tmp:
        root = Path(tmp)
        config = config_data(model)
        config["engine"]["runtime_adapters"] = gpu
        config.pop("world_size")
        config["nodes"] = ["127.0.0.1"] * count
        config["ports"] = {"fabric": free_port(), "journal": free_port()}
        path = root / "cluster.json"
        path.write_text(json.dumps(config))
        parent, child = socket.socketpair()
        parent.settimeout(60)
        processes, logs = [], []
        env = os.environ if gpu else {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        try:
            for rank in range(count):
                config_path = path
                if mismatch == "settings" and rank == 1:
                    bad = {**config, "engine": {**config["engine"], "default_max_tokens": 7}}
                    config_path = root / "bad.json"
                    config_path.write_text(json.dumps(bad))
                log = (root / f"r{rank}.log").open("wb")
                logs.append(log)
                rank_binary = binary
                rank_env = env
                if mismatch == "environment" and rank == 1:
                    rank_env = {**env, "DGPP_HERETIC_TEST_MISMATCH": "1"}
                if mismatch == "binary" and rank == 1:
                    rank_binary = str(root / "different-worker")
                    shutil.copy2(binary, rank_binary)
                    with open(rank_binary, "ab") as f:
                        f.write(b"different-build")
                argv = [rank_binary, "--config", str(config_path), "--rank", str(rank), "--op-log", str(root / f"r{rank}.ops")]
                if rank == 0:
                    argv += ["--control-fd", str(child.fileno())]
                processes.append(subprocess.Popen(argv, stdout=log, stderr=log, env=rank_env,
                    pass_fds=(child.fileno(),) if rank == 0 else ()))
            child.close()
            yield Client(parent), processes, root
        except BaseException:
            for rank in range(len(processes)):
                print((root / f"r{rank}.log").read_text()[-12000:], file=sys.stderr)
            raise
        finally:
            parent.close()
            child.close()
            for p in processes:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait(timeout=5)
            for log in logs:
                log.close()


def host(binary, metadata):
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp)
        (model / "config.json").write_text(Path(metadata).read_text())
        for count in (2, 4):
            with world(binary, model, count, False) as (client, processes, root):
                desc = client.call("describe")[0]
                assert desc["catalog"]["world"] == count
                client.call("close")
                assert all(p.wait(timeout=5) == 0 for p in processes)
                assert len({(root / f"r{r}.ops").read_bytes() for r in range(count)}) == 1
            for mismatch in ("settings", "environment", "binary"):
                with world(binary, model, count, False, mismatch=mismatch) as (_, processes, root):
                    assert all(p.wait(timeout=10) != 0 for p in processes)
                    assert "mismatch on rank 1" in (root / "r0.log").read_text()
        with world(binary, model, 2, False) as (client, processes, _):
            client.call("describe")
            processes[1].kill()
            processes[1].wait(timeout=5)
            try:
                client.call("describe")
            except (EOFError, ConnectionError):
                pass
            else:
                raise AssertionError("a short world returned a result")
            assert processes[0].wait(timeout=5) != 0
    print("distributed host: worlds 2/4, matching audit streams, settings mismatch and peer death passed")


def gpu(binary, fixture):
    entries = read_safetensors_index(fixture)
    for count in (2, 4):
        with world(binary, fixture, count, True) as (client, processes, root):
            desc = client.call("describe")[0]
            targets = desc["catalog"]["targets"]
            weights = {t["id"]: load_linear(entries, t["module"], t["output_features"], t["input_features"]) for t in targets}
            check_weight_reads(client, targets, weights)
            for length in (1, 3, 16, 33):
                ids = [(i * 17 + 11) % desc["vocab_size"] for i in range(length)]
                _, base, base_bytes = client.call("first_token_logits", token_ids=ids)
                _, capture, capture_bytes = client.call("capture", token_ids=ids)
                assert len(base["logits"]) == desc["vocab_size"]
                assert base_bytes["logits"] == capture_bytes["logits"], "TP capture changed logits"
                for target in targets:
                    x, y = capture[target["id"] + ".input"], capture[target["id"] + ".output"]
                    assert len(x) == target["input_features"] and len(y) == target["output_features"]
                    want = [sum(a * b for a, b in zip(x, row)) for row in weights[target["id"]]]
                    relative = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, want)) / max(sum(b * b for b in want), 1e-30))
                    limit = 2e-5 if "shared_experts" in target["module"] else 0.008
                    assert relative < limit, (count, length, target["id"], relative)
                generated = client.call("generate", token_ids=ids, max_new_tokens=4, stop_token_ids=[])[1]["token_ids"]
                assert generated[0] == max(range(desc["vocab_size"]), key=base["logits"].__getitem__)
                assert client.call("generate", token_ids=ids, max_new_tokens=4, stop_token_ids=[])[1]["token_ids"] == generated
                assert client.call("generate", token_ids=ids, max_new_tokens=4, stop_token_ids=[generated[0]])[1]["token_ids"] == generated[:1]
                assert client.call("first_token_logits", token_ids=ids)[2] == base_bytes
                print(f"TP{count} {Path(fixture).name} tokens={length}: full logits, captures, projection oracle, generation/reset passed")
            check_runtime(client, desc)
            client.call("close")
            assert all(p.wait(timeout=10) == 0 for p in processes)
            assert len({(root / f"r{r}.ops").read_bytes() for r in range(count)}) == 1, "operation streams diverged"


if __name__ == "__main__":
    mode, binary, fixture = sys.argv[1:]
    (host if mode == "host" else gpu)(binary, fixture)
