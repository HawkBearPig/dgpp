"""Read-only deployment preflight checks, locally and over SSH.

The probe is a standalone Python module: the coordinator sends this file to
the remote Python interpreter's stdin, without installing files or sourcing
.env. Only the resolved, allowlisted node settings are sent to a peer.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import struct
import subprocess
import sys


def require_head(host):
    """Reject a coordinator command run on a machine other than rank 0."""
    try:
        with socket.socket() as connection:
            connection.bind((socket.gethostbyname(host), 0))
    except OSError as error:
        raise ValueError("run this command on rank 0 (the first address in DGPP_NODES)") from error


def cache_root(env=None):
    env = os.environ if env is None else env
    return Path(env.get("HF_HUB_CACHE") or
                (str(Path(env["HF_HOME"]) / "hub") if env.get("HF_HOME") else
                 "~/.cache/huggingface/hub")).expanduser()


def cached_snapshot(model, root):
    if len(model.split("/")) != 2 or any(part in ("", ".", "..") for part in model.split("/")):
        raise ValueError("model must be a Hugging Face ORG/NAME repository ID")
    repository = Path(root) / ("models--" + model.replace("/", "--"))
    ref = repository / "refs/main"
    if ref.is_file():
        revision = ref.read_text().strip()
        if not revision or "/" in revision or revision in (".", ".."):
            raise ValueError("invalid refs/main in checkpoint cache")
        snapshot = repository / "snapshots" / revision
    else:
        snapshots = sorted((repository / "snapshots").glob("*"))
        snapshots = [path for path in snapshots if path.is_dir()]
        if len(snapshots) != 1:
            raise ValueError(f"{model}: need refs/main or exactly one cached snapshot under {repository}")
        snapshot = snapshots[0]
    if not snapshot.is_dir():
        raise ValueError(f"cached snapshot is missing: {snapshot}")
    return snapshot


def checkpoint_size(snapshot):
    """Validate required metadata and safetensors lengths without reading weights."""
    snapshot = Path(snapshot)
    for name in ("config.json", "tokenizer.json"):
        if not isinstance(json.loads((snapshot / name).read_text()), dict):
            raise ValueError(f"{name} must contain a JSON object")
    # The prompt format: a Jinja template, or a DeepSeek release's own encoder
    # (V4.1's `encoding/encoding.py`, V4's `encoding/encoding_dsv4.py`; those
    # checkpoints ship no template).
    encoders = ("encoding.py", "encoding_dsv4.py")
    if not (snapshot / "chat_template.jinja").is_file() and not any(
        (snapshot / "encoding" / name).is_file() for name in encoders
    ):
        raise ValueError(f"missing chat_template.jinja (or encoding/encoding.py, encoding/encoding_dsv4.py) in {snapshot}")
    return shard_bytes(snapshot)


def shard_bytes(snapshot):
    """Validate the safetensors shards' headers and lengths (the index's, or
    every shard in the directory — the Qwen n-gram table repository) and
    return their indexed bytes."""
    snapshot = Path(snapshot)
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        metadata = json.loads(index.read_text())
        weight_map = metadata.get("weight_map", {}) if isinstance(metadata, dict) else None
        if not isinstance(weight_map, dict) or not weight_map or any(not isinstance(v, str) for v in weight_map.values()):
            raise ValueError("checkpoint index has no weight_map")
        names = sorted(set(weight_map.values()))
    elif (snapshot / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        names = sorted(p.name for p in snapshot.glob("*.safetensors"))
        if not names:  # a missing checkpoint, as a missing shard is: OSError
            raise FileNotFoundError(f"no safetensors shards in {snapshot}")
    total = 0
    for name in names:
        if (not isinstance(name, str) or not name or "\\" in name or
                any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("checkpoint index contains an invalid shard path")
        path = snapshot / name
        with path.open("rb") as shard:
            size = path.stat().st_size
            prefix = shard.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated shard: {name}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= min(size - 8, 100_000_000):
                raise ValueError(f"invalid safetensors header: {name}")
            header = json.loads(shard.read(header_size))
        if not isinstance(header, dict):
            raise ValueError(f"invalid safetensors header: {name}")
        ends = []
        for tensor, metadata in header.items():
            if tensor == "__metadata__":
                continue
            if not isinstance(metadata, dict) or not isinstance(metadata.get("data_offsets"), list) or len(metadata["data_offsets"]) != 2:
                raise ValueError(f"invalid tensor metadata: {name}/{tensor}")
            start, end = metadata["data_offsets"]
            if type(start) is not int or type(end) is not int or start < 0 or end < start:
                raise ValueError(f"invalid tensor offsets: {name}/{tensor}")
            ends.append(end)
        if not ends or max(ends) + 8 + header_size != size:
            raise ValueError(f"incomplete or inconsistent shard length: {name}")
        if index.is_file():
            absent = [tensor for tensor, shard in weight_map.items() if shard == name and tensor not in header]
            if absent:
                raise ValueError(f"index references missing tensor in {name}: {absent[0]}")
        total += size
    return total


def command(argv, timeout=20, env=None):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def server_version(binary, env=None):
    """Apply the process swap policy to metadata probes as well as servers."""
    no_swap = (os.environ if env is None else env).get("DGPP_NO_SWAP", "0")
    if no_swap not in ("0", "1"):
        raise ValueError("DGPP_NO_SWAP must be 0 or 1")
    argv = [binary, "--version"]
    if no_swap == "1":
        # This module is sent alone to peer interpreters, so it cannot import
        # the process-control module used by long-running rank launches.
        argv = ["systemd-run", "--user", "--scope", "--quiet",
                "--property=MemorySwapMax=0", "--", *argv]
    return command(argv, env=env)


def ancestor(path):
    path = Path(path).expanduser().absolute()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def active_roce_port(port):
    try:
        return ((port / "state").read_text().startswith("4:") and
                (port / "link_layer").read_text().strip() == "Ethernet")
    except OSError:
        return False


def roce_gids(port):
    result = []
    for path in sorted((port / "gid_attrs/types").glob("*")):
        try:
            if "v2" in path.read_text().lower():
                result.append((path.name, (port / "gids" / path.name).read_text().strip()))
        except OSError:
            # Drivers expose unused GID table slots whose reads return EINVAL.
            continue
    return result


def roce_inventory(root=Path("/sys/class/infiniband")):
    """Map verbs ports and GIDs to Linux interfaces without opening any QPs."""
    addresses = {}
    try:
        result = command(["ip", "-j", "address"])
        if result.returncode == 0:
            addresses = {item["ifname"]: item for item in json.loads(result.stdout)}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    rows = []
    for device in sorted(root.glob("*")):
        for port in sorted((device / "ports").glob("*")):
            gids = []
            for index, value in roce_gids(port):
                try:
                    address = ipaddress.IPv6Address(value)
                except ValueError:
                    continue
                if address.is_link_local or address.is_unspecified:
                    continue
                try:
                    interface = (port / "gid_attrs/ndevs" / index).read_text().strip()
                except OSError:
                    interface = ""
                network = addresses.get(interface, {})
                gids.append({"index": int(index), "address": str(address.ipv4_mapped or address),
                             "interface": interface, "mtu": network.get("mtu"),
                             "ips": [f"{a['local']}/{a['prefixlen']}" for a in network.get("addr_info", [])]})
            rows.append({"device": device.name, "port": port.name,
                         "active": active_roce_port(port), "gids": gids,
                         "usable": port.name == "1" and active_roce_port(port) and bool(gids)})
    return rows


def probe(spec):
    checks = []
    def record(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})
    env = dict(os.environ)
    env.update({key: os.path.expanduser(value) for key, value in spec["env"].items()})
    record("platform", "ok" if platform.system() == "Linux" and platform.machine() == "aarch64" else "fail",
           f"{platform.system()} {platform.machine()}; supported target is Linux/aarch64 GB10")
    record("Python", "ok" if sys.version_info >= (3, 10) else "fail",
           f"{platform.python_version()}; Python 3.10+ is required")
    required = ["python3", "timeout", "ldd"]
    if spec["rank"] == 0:
        required += ["ssh", "scp", "curl", "jq"]
    if spec.get("preparing"):
        required += ["ip"]
        if len(spec["nodes"]) > 1:
            required += ["rsync"]
        if spec["rank"] == 0:
            required += ["pdftotext"]
    missing = [name for name in required if not shutil.which(name)]
    record("commands", "fail" if missing else "ok", "missing: " + ", ".join(missing) if missing else ", ".join(required))
    try:
        address = socket.gethostbyname(spec["nodes"][spec["rank"]])
        with socket.socket() as connection:
            connection.bind((address, 0))
        record("node address", "ok", f"rank {spec['rank']} address {address} belongs to this host")
    except OSError as error:
        record("node address", "fail", f"run the launcher on DGPP_NODES[0]; check rank order and DNS: {error}")
    try:
        gpu = command(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"])
        record("GPU", "ok" if gpu.returncode == 0 and any("12.1" in line for line in gpu.stdout.splitlines()) else "fail",
               gpu.stdout.strip() or gpu.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        record("GPU", "fail", str(error))
    for name, path in {**spec["paths"], "model cache": str(cache_root(env))}.items():
        if not path:
            continue
        parent = ancestor(path)
        # Serving only READS the checkpoint cache (a read-only model store is a
        # legitimate site layout, and `up` runs this preflight); preparation
        # downloads and syncs into it. Every other path here is written at run time.
        writes = name != "model cache" or bool(spec.get("preparing"))
        usable = os.access(parent, (os.W_OK if writes else os.R_OK) | os.X_OK)
        free = shutil.disk_usage(parent).free
        record(name, "ok" if usable else "fail", f"{path}: parent {parent}, {free / 2**30:.1f} GiB free" +
               ("" if usable else "; not writable" if writes else "; not readable"))
    if not spec.get("preparing"):
        try:
            snapshot = cached_snapshot(spec["model"], cache_root(env))
            size = checkpoint_size(snapshot)
            record("checkpoint", "ok", f"{snapshot}: {size / 2**30:.1f} GiB; all indexed shards have consistent lengths (not a content hash check)")
        except (OSError, ValueError, KeyError, TypeError) as error:
            record("checkpoint", "fail", f"checkpoint missing, incomplete or ambiguous: {error}; "
                   "on rank 0 run python3 scripts/download_model.py --config FILE "
                   "with this deployment's filename; add --sync-only if rank 0 already has the complete checkpoint")
        # The Qwen n-gram table from another snapshot (engine.ngram_table_model):
        # its shards' headers and lengths, nothing else is required of it.
        if spec.get("table_model"):
            try:
                table = cached_snapshot(spec["table_model"], cache_root(env))
                tsize = shard_bytes(table)
                record("n-gram table", "ok", f"{table}: {tsize / 2**30:.1f} GiB of shards")
            except (OSError, ValueError, KeyError, TypeError) as error:
                record("n-gram table", "fail", f"the table snapshot is missing, incomplete or ambiguous: {error}; "
                       "python3 scripts/download_model.py --config FILE downloads it beside the model")
    try:
        memory = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
        record("available memory", "ok", f"{int(memory.split()[1]) / 2**20:.1f} GiB; startup enforces the exact model memory plan")
    except (OSError, StopIteration, ValueError) as error:
        record("available memory", "fail", str(error))
    ports = []
    if spec["rank"] == 0:
        ports.append((spec["http_bind"], spec["ports"]["http"], "HTTP"))
        if len(spec["nodes"]) > 1:
            ports.extend(("0.0.0.0", spec["ports"][name], name) for name in ("fabric", "journal"))
    for host, port, name in ports:
        try:
            with socket.socket() as connection:
                # The probe binds as the servers do (src/net/tcp.cpp and
                # src/serve/http_server.cpp set SO_REUSEADDR): a listener's
                # TIME_WAIT connections from the previous world are not a
                # conflict, only a live listener is. Without it a restart
                # within a minute of a stop failed here (2026-09-13, the
                # failure drill's reboot).
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                connection.bind((host, port))
            record(name + " port", "ok", f"{host}:{port} is available")
        except OSError as error:
            record(name + " port", "fail", f"{host}:{port}: {error}; do not stop unrelated services to free it")
    devices = env.get("DGPP_ROCE_DEVICES", "").split()
    if len(spec["nodes"]) > 1:
        if not devices:
            devices = sorted(path.name for path in Path("/sys/class/infiniband").glob("*")
                             if active_roce_port(path / "ports/1"))
            record("RoCE selection", "warn", "auto-selected active devices; set DGPP_ROCE_DEVICES explicitly on multi-fabric hosts")
        if not devices:
            record("RoCE devices", "fail", "no active RoCE devices; check rdma-core, cabling and network setup")
        if len(devices) > 2:
            record("RoCE devices", "fail", "the transport supports at most two lanes; select DGPP_ROCE_DEVICES explicitly")
        indices = env.get("DGPP_ROCE_GID_INDICES", "").split()
        for index, device in enumerate(devices):
            ports_dir = Path("/sys/class/infiniband") / device / "ports"
            active = [p for p in ports_dir.glob("1") if active_roce_port(p)]
            if not active:
                record("RoCE " + device, "fail", "device port 1 is not active Ethernet")
                continue
            selected = active[0]
            gids = roce_gids(selected)
            if indices:
                gids = [(i, gid) for i, gid in gids if i == indices[index]]
            routable = [(i, gid) for i, gid in gids if not gid.startswith("fe80:") and gid != "0000:0000:0000:0000:0000:0000:0000:0000"]
            record("RoCE " + device, "ok" if routable else "fail",
                   f"port {selected.name}, routable v2 GIDs: {routable}; verify these subnets connect corresponding lanes on every node")
    binary = spec.get("binary")
    if binary:
        binary = os.path.expanduser(binary)
        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            record("server binary", "fail", f"missing executable {binary}; build dgpp_serve_app with libibverbs enabled")
        else:
            try:
                libraries = command(["ldd", binary])
                record("runtime libraries", "fail" if libraries.returncode or "not found" in libraries.stdout else "ok",
                       libraries.stdout.strip() if libraries.returncode or "not found" in libraries.stdout else "all binary dependencies resolve")
                version = server_version(binary, env=env)
                record("server version", "ok" if version.returncode == 0 else "fail", version.stdout.strip() or version.stderr.strip())
            except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                record("server binary", "fail", str(error))
    else:
        # Development launches stage the binary later. Check the peer's loader
        # cache now, but do not pretend this verifies the staged binary's ABI.
        try:
            libraries = command(["ldconfig", "-p"])
            needed = ["libibverbs.so", "libcudart.so.13", "libcublasLt.so.13", "libstdc++.so"]
            absent = [name for name in needed if name not in libraries.stdout]
            record("runtime libraries" if spec["rank"] == 0 else "peer runtime libraries",
                   "fail" if absent else "ok", "missing: " + ", ".join(absent) if absent else ", ".join(needed))
            if not spec.get("preparing"):
                record("peer binary", "warn", "development binary will be staged; verify identical OS/compiler runtime compatibility")
        except (OSError, subprocess.TimeoutExpired) as error:
            record("peer runtime libraries", "fail", str(error))
    return {"rank": spec["rank"], "checks": checks, "devices": devices}


def check_cluster(cfg, binary, log_dir, stage_dir, user, peer_binary=None, *, local_only=False, preparing=False):
    reports = []
    for rank, host in enumerate(cfg["nodes"]):
        if local_only and rank:
            continue
        probe_env = dict(cfg["node_env"][rank])
        if "DGPP_NO_SWAP" in os.environ and "DGPP_NO_SWAP" not in probe_env:
            probe_env["DGPP_NO_SWAP"] = os.environ["DGPP_NO_SWAP"]
        spec = {"rank": rank, "nodes": cfg["nodes"], "model": cfg["model"],
                "table_model": (cfg.get("engine", {}) or {}).get("ngram_table_model"),
                "ports": cfg["ports"], "http_bind": cfg["http"]["bind_host"],
                "env": probe_env, "binary": binary if rank == 0 else peer_binary,
                "preparing": preparing,
                "paths": {"logs/staging": log_dir if rank == 0 else stage_dir,
                          "resident cache": cfg["node_env"][rank].get("DGPP_RESIDENT_CACHE_DIR") or
                                            cfg["paths"].get("resident_cache") or "~/.cache/dgpp/resident"}}
        try:
            if rank == 0:
                report = probe(spec)
            else:
                remote = "python3 - probe --spec " + shlex.quote(json.dumps(spec))
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                                         f"{user}@{host}" if user else host, remote],
                                        input=Path(__file__).read_text(), capture_output=True, text=True, timeout=120)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or "remote probe failed")
                report = json.loads(result.stdout)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            report = {"rank": rank, "checks": [{"check": "probe", "status": "fail", "detail": str(error)}], "devices": None}
        reports.append(report)
        for check in report["checks"]:
            print(f"{check['status'].upper():4} rank {rank} ({host}) {check['check']}: {check['detail']}")
    failures = sum(check["status"] == "fail" for report in reports for check in report["checks"])
    inspected = [report for report in reports if report["devices"] is not None]
    if len({len(report["devices"]) for report in inspected}) > 1:
        print("FAIL RoCE lane counts differ between successfully inspected nodes")
        failures += 1
    if len(inspected) != len(reports):
        print("WARN RoCE lane counts are unknown on nodes whose probes failed; fix those probe errors first.")
    print(f"Preflight: {failures} failed check(s). No files were installed and no services were started or stopped.")
    if local_only:
        print("Remote nodes were not checked (--local-only).")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "cache", "roce"))
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec)
    if args.action == "roce":
        print(json.dumps(roce_inventory()))
    elif args.action == "cache":
        root = cache_root({**os.environ, **spec["env"]}).absolute()
        result = {"root": str(root), "repository": str(root / ("models--" + spec["model"].replace("/", "--")))}
        if spec.get("verify"):
            snapshot = cached_snapshot(spec["model"], root)
            result.update(snapshot=str(snapshot), size=checkpoint_size(snapshot))
            if spec.get("revision") and snapshot.name != spec["revision"]:
                parser.exit(2, "active peer revision differs from the head; run download_model.py --config FILE --sync-only\n")
            if spec.get("table_model"):
                table = cached_snapshot(spec["table_model"], root)
                result.update(table_snapshot=str(table), table_size=shard_bytes(table))
        print(json.dumps(result))
    else:
        print(json.dumps(probe(spec)))
