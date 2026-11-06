"""Native worker wire, bounds, lifecycle, and fixture projection oracle.

Host mode needs only model config metadata and runs with CUDA hidden. GPU mode
uses the tiny fixture from heretic_capture_test and pure-Python weight math.
"""
import contextlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from glm_dsa_reference_dump import bf16_from_bits, load_linear, load_bf16, read_safetensors_index


def read(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("worker closed")
        data.extend(chunk)
    return data


class Client:
    def __init__(self, sock):
        self.sock = sock
        self.seq = 0

    def request(self, op, **extra):
        self.seq += 1
        return dict(schema="dgpp.heretic.rpc", version=1, seq=self.seq, op=op, **extra)

    def call(self, op, **extra):
        request = self.request(op, **extra)
        data = json.dumps(request).encode()
        self.sock.sendall(struct.pack("<II", len(data), 0) + data)
        nh, np = struct.unpack("<II", read(self.sock, 8))
        assert 0 < nh <= 65536 and np <= 64 * 1024 * 1024
        header, payload = json.loads(read(self.sock, nh)), read(self.sock, np)
        assert header["seq"] == self.seq and header["op"] == op and header["state"] == "baseline"
        values, raw = {}, {}
        offset = 0
        for t in header["tensors"]:
            name, size = t["name"], t["nbytes"]
            assert t["offset"] == offset and offset + size <= len(payload) and name not in values
            raw[name] = payload[offset:offset + size]
            count = math.prod(t["shape"])
            code = {"bfloat16": "H", "float32": "f", "int64": "q"}[t["dtype"]]
            assert count * struct.calcsize(code) == size
            values[name] = list(struct.unpack(f"<{count}{code}", raw[name]))
            if code == "H":
                values[name] = [bf16_from_bits(v) for v in values[name]]
            offset += size
        assert offset == len(payload)
        return header, values, raw


@contextlib.contextmanager
def worker(binary, config, gpu=False):
    parent, child = socket.socketpair()
    parent.settimeout(30)
    with tempfile.TemporaryFile() as logs:
        process = subprocess.Popen([binary, "--config", str(config), "--control-fd", str(child.fileno())],
            pass_fds=(child.fileno(),), stdout=logs, stderr=logs,
            env=os.environ if gpu else {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        child.close()
        try:
            yield Client(parent), process, logs
        finally:
            parent.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def config_data(model):
    return {"model": str(model), "world_size": 1, "engine": {
        "heretic_research": True, "max_concurrency": 1, "prefix_cache_gib": 0,
        "kv_capacity": 128, "default_max_tokens": 8}}


def host(binary, metadata):
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        (model / "config.json").write_text(Path(metadata).read_text())
        path = Path(tmp) / "cluster.json"
        config = config_data(model)
        path.write_text(json.dumps(config))
        with worker(binary, path) as (client, process, _):
            d, values, _ = client.call("describe")
            assert d["catalog"]["world"] == 1 and len(d["catalog"]["targets"]) == 10
            assert "capture_last_token" in d["capabilities"] and not values
            client.call("close")
            assert process.wait(timeout=5) == 0
        # A disconnected owner is also a clean, reaped exit.
        with worker(binary, path) as (client, process, _):
            client.call("describe")
            client.sock.close()
            assert process.wait(timeout=5) == 0
        cases = [
            (lambda c: c["engine"].update(heretic_research=False), "research is disabled"),
            (lambda c: (c.pop("world_size"), c.update(nodes=["rank0", "rank1", "rank2"])), "world size 1, 2 or 4"),
            (lambda c: c["engine"].update(mtp=True), "mtp=false"),
            (lambda c: c["engine"].update(max_concurrency=2), "max_concurrency=1"),
            (lambda c: c["engine"].update(kv_dtype="fp8"), "BF16 KV"),
            (lambda c: c["engine"].update(prefill_fold_scales=True), "unknown or duplicate"),
        ]
        for change, error in cases:
            bad = config_data(model)
            change(bad)
            if "research is disabled" == error:
                bad["model"] = "/nonexistent-model"
            path.write_text(json.dumps(bad))
            result = subprocess.run([binary, "--config", str(path), "--control-fd", "9999"],
                capture_output=True, text=True, timeout=10, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            assert result.returncode != 0 and error in result.stderr, result.stderr
        path.write_text(json.dumps(config))
        malformed = [
            (struct.pack("<II", 65537, 0), "frame lengths"),
            (struct.pack("<II", 1, 1), "frame lengths"),
            ({"op": "capture", "token_ids": []}, "prompt token count"),
            ({"op": "capture", "token_ids": [True]}, "integer"),
            ({"op": "capture", "token_ids": [-1]}, "token ID"),
            ({"op": "capture", "token_ids": [1] * 129}, "prompt token count"),
            ({"op": "generate", "token_ids": [1], "max_new_tokens": 9, "stop_token_ids": []}, "token budget"),
            ({"op": "generate", "token_ids": [1], "max_new_tokens": 1, "stop_token_ids": [999999]}, "stop token"),
            ({"op": "install_adapter"}, "unsupported operation"),
            ({"op": "describe", "seq": 2}, "sequence mismatch"),
            ({"op": "describe", "version": 2}, "version/schema"),
            ({"op": "describe", "surprise": True}, "unknown or duplicate"),
        ]
        for bad, error in malformed:
            with worker(binary, path) as (client, process, logs):
                if isinstance(bad, dict):
                    request = client.request("describe")
                    request.update(bad)
                    data = json.dumps(request).encode()
                    packet = struct.pack("<II", len(data), 0) + data
                else:
                    packet = bad
                client.sock.sendall(packet)
                assert process.wait(timeout=5) != 0
                logs.seek(0)
                assert error in logs.read().decode(), error
        print("worker host: opt-in, no CUDA, capabilities, bounds, sequences, fail-closed and owner cleanup passed")


def gpu(binary, model):
    entries = read_safetensors_index(model)
    oracle = json.loads((Path(model) / "worker_oracle.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cluster.json"
        path.write_text(json.dumps(config_data(model)))
        with worker(binary, path, gpu=True) as (client, process, logs):
            try:
                description = client.call("describe")[0]
                targets = description["catalog"]["targets"]
                weights = {t["id"]: load_linear(entries, t["module"], t["output_features"], t["input_features"]) for t in targets}
                embed_shape, embeddings = load_bf16(entries, "model.embed_tokens.weight")
                for length in (1, 3, 16, 33):
                    ids = [(i * 17 + 11) % description["vocab_size"] for i in range(length)]
                    baseline = client.call("first_token_logits", token_ids=ids)
                    _, captured, raw = client.call("capture", token_ids=ids)
                    assert raw["logits"] == baseline[2]["logits"], "worker capture changed logits"
                    h = description["hidden_size"]
                    assert captured["residuals"][:h] == embeddings[ids[-1] * h:(ids[-1] + 1) * h]
                    for t in targets:
                        x, y = captured[t["id"] + ".input"], captured[t["id"] + ".output"]
                        want = [sum(a * b for a, b in zip(x, row)) for row in weights[t["id"]]]
                        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, want)) / max(sum(b * b for b in want), 1e-30))
                        # Shared projections expose FP32 accumulators; the other
                        # paths have one BF16 output rounding (~0.2% relative).
                        tolerance = 2e-5 if "shared_experts" in t["module"] else 0.004
                        assert err < tolerance, (length, t["id"], err)
                    generated = client.call("generate", token_ids=ids, max_new_tokens=4, stop_token_ids=[])[1]["token_ids"]
                    assert generated == oracle[str(length)], "worker decode differs from direct session API"
                    assert generated[0] == max(range(len(baseline[1]["logits"])), key=baseline[1]["logits"].__getitem__)
                    stopped = client.call("generate", token_ids=ids, max_new_tokens=4, stop_token_ids=[generated[0]])
                    assert stopped[0]["stop_reason"] == "stop_token" and stopped[1]["token_ids"] == generated[:1]
                    assert client.call("first_token_logits", token_ids=ids)[2] == baseline[2]
                    print(f"worker fixture {Path(model).name}, tokens={length}: exact logits, projection oracle, generation and reset passed")
                client.call("close")
                assert process.wait(timeout=10) == 0
            except BaseException:
                logs.seek(0)
                print(logs.read().decode(), file=sys.stderr)
                raise


if __name__ == "__main__":
    mode, binary, fixture = sys.argv[1:]
    (host if mode == "host" else gpu)(binary, fixture)
