"""Shared site settings. Parse .env as data; exported settings take precedence."""
import argparse
import getpass
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
SITE_KEYS = (
    "DGPP_NODES", "DGPP_SSH_USER", "DGPP_CLUSTER_CONFIG",
    "DGPP_HTTP_PORT", "DGPP_FABRIC_PORT", "DGPP_JOURNAL_PORT",
    "DGPP_LOG_DIR", "DGPP_STAGE_DIR", "DGPP_RELEASE_DIR",
    "DGPP_HTTP_BIND", "DGPP_ROCE_DEVICES", "DGPP_ROCE_GID_INDICES",
    "HF_HOME", "HF_HUB_CACHE", "DGPP_RESIDENT_CACHE_DIR", "DGPP_NODE_OVERRIDES",
    "DGPP_BUILD_DIR", "DGPP_DATA_DIR",
    "DGPP_LOG_LEVEL", "DGPP_MLOCK",
    # The engine's L2 weight-prefetch knobs (src/kernels/l2_prefetch.hpp); site
    # settings so an A/B runs with the same setting on every rank.
    "DGPP_L2_PREFETCH", "DGPP_L2_PREFETCH_MB", "DGPP_L2_PREFETCH_BOUNDARY", "DGPP_L2_PREFETCH_LAYER",
    # The bus timeline (DGPP_BUS_TIMELINE=1: the graph windows' and the prefill
    # folds' decomposition at INFO) and the dense-lowering A/B switches
    # (DGPP_DSV41_DENSE_GEMV=1 for DeepSeek; DGPP_DENSE_GEMV_ROWS=n for the
    # session-core families, kernels/gemm.hpp): every rank the same, or the
    # ranks' walks differ.
    "DGPP_BUS_TIMELINE", "DGPP_DSV41_DENSE_GEMV", "DGPP_DENSE_GEMV_ROWS", "DGPP_DSV41_EAGER_FOLD",
)
NODE_KEYS = ("DGPP_ROCE_DEVICES", "DGPP_ROCE_GID_INDICES", "HF_HUB_CACHE", "DGPP_RESIDENT_CACHE_DIR",
    "DGPP_LOG_LEVEL", "DGPP_MLOCK",
    # The engine's L2 weight-prefetch knobs (src/kernels/l2_prefetch.hpp): an A/B runs
    # with the same setting on every rank.
    "DGPP_L2_PREFETCH", "DGPP_L2_PREFETCH_MB", "DGPP_L2_PREFETCH_BOUNDARY", "DGPP_L2_PREFETCH_LAYER",
    "DGPP_BUS_TIMELINE", "DGPP_DSV41_DENSE_GEMV", "DGPP_DENSE_GEMV_ROWS", "DGPP_DSV41_EAGER_FOLD",
)
DEFAULTS = {
    "DGPP_HTTP_PORT": "18080", "DGPP_FABRIC_PORT": "29970",
    "DGPP_JOURNAL_PORT": "29971", "DGPP_LOG_DIR": "~/dgpp/log",
    "DGPP_STAGE_DIR": "/tmp/bus4", "DGPP_RELEASE_DIR": "~/dgpp/releases",
    "DGPP_HTTP_BIND": "127.0.0.1",
}


def config_argument(value):
    """Do not let an unset shell variable select a different deployment."""
    if not value.strip():
        raise argparse.ArgumentTypeError("--config must not be empty; set CONFIG in this shell or pass the deployment filename")
    return value


def read_env(path):
    """Read allowlisted KEY=value assignments, optionally quoted.

    Values are single-line literals, with no interpolation or shell execution.
    Unrelated entries, including credentials, are not returned.
    """
    values = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if key not in SITE_KEYS:
            continue
        if not separator:
            raise ValueError(f"{path}:{number}: expected {key}=value")
        value = value.strip()
        if value.startswith(("'", '"')):
            end = value.find(value[0], 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith("#")):
                raise ValueError(f"{path}:{number}: invalid quoted value for {key}")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if "\0" in value:
            raise ValueError(f"{path}:{number}: invalid value for {key}")
        if key in values:
            raise ValueError(f"{path}:{number}: duplicate {key}")
        values[key] = value
    return values


def settings(environ=None):
    environ = os.environ if environ is None else environ
    path = Path(environ.get("DGPP_ENV_FILE", str(ROOT / ".env"))).expanduser()
    if path.exists():
        values = read_env(path)
    elif "DGPP_ENV_FILE" in environ:
        raise ValueError(f"site environment file does not exist: {path}")
    else:
        values = {}
    result = dict(DEFAULTS)
    result.update(values)
    result.update({key: environ[key] for key in SITE_KEYS if key in environ})
    for key in ("DGPP_BUILD_DIR", "DGPP_DATA_DIR"):
        if result.get(key):
            path = Path(result[key]).expanduser()
            result[key] = str(path if path.is_absolute() else ROOT / path)
    return result


def config_path(values=None):
    values = settings() if values is None else values
    path = Path(values.get("DGPP_CLUSTER_CONFIG") or "deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json").expanduser()
    return str((path if path.is_absolute() else ROOT / path).resolve())


def site_nodes(values):
    nodes = values.get("DGPP_NODES", "").split()
    if not nodes:
        raise ValueError("set DGPP_NODES in .env to the node addresses in rank order")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", node) for node in nodes):
        raise ValueError("DGPP_NODES must contain space-separated IP addresses or hostnames")
    if len(set(nodes)) != len(nodes):
        raise ValueError("DGPP_NODES must not contain duplicate nodes")
    return nodes


def ssh_user(values):
    user = values.get("DGPP_SSH_USER") or getpass.getuser()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\$?", user):
        raise ValueError("DGPP_SSH_USER is not a valid SSH login name")
    return user


def port(values, name):
    value = values[f"DGPP_{name.upper()}_PORT"]
    if not value.isdigit() or not 1 <= int(value) <= 65535:
        raise ValueError(f"DGPP_{name.upper()}_PORT must be an integer in [1, 65535]")
    return int(value)


def site_paths(values):
    result = {name: values[f"DGPP_{name.upper()}"] for name in ("log_dir", "stage_dir", "release_dir")}
    for name in ("stage_dir", "release_dir"):
        value = result[name]
        if (not re.fullmatch(r"(?:/|~/)[A-Za-z0-9_./+@=-]+", value)
                or ".." in value.split("/") or not value.strip("~/.")):
            raise ValueError(f"DGPP_{name.upper()} must be an absolute or ~/ directory without spaces or shell syntax")
    if not result["log_dir"]:
        raise ValueError("DGPP_LOG_DIR must not be empty")
    return result


def deployment(path):
    cfg = json.loads(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("deployment config must be an object")
    if set(cfg) & {"nodes", "ssh_user", "ports"}:
        raise ValueError("move nodes, ssh_user, and ports out of the deployment JSON into .env; use world_size")
    unknown = set(cfg) - {"model", "world_size", "release", "engine", "paths", "http", "nvme_cache"}
    if unknown:
        raise ValueError("unknown deployment keys: " + ", ".join(sorted(unknown)))
    nvme = cfg.get("nvme_cache", {})
    if not isinstance(nvme, dict) or set(nvme) - {"enabled", "path", "capacity_gib", "min_tokens"}:
        raise ValueError("nvme_cache may only contain enabled, path, capacity_gib and min_tokens")
    if "enabled" in nvme and not isinstance(nvme["enabled"], bool):
        raise ValueError("nvme_cache.enabled must be true or false")
    if "path" in nvme and (not isinstance(nvme["path"], str) or not nvme["path"]):
        raise ValueError("nvme_cache.path must be a non-empty directory path")
    if "capacity_gib" in nvme and (isinstance(nvme["capacity_gib"], bool)
                                  or not isinstance(nvme["capacity_gib"], (int, float))
                                  or nvme["capacity_gib"] < 0):
        raise ValueError("nvme_cache.capacity_gib must be a non-negative number of GiB")
    if "min_tokens" in nvme and (type(nvme["min_tokens"]) is not int or nvme["min_tokens"] < 0):
        raise ValueError("nvme_cache.min_tokens must be a non-negative integer")
    if nvme.get("enabled") and not nvme.get("capacity_gib"):
        raise ValueError("nvme_cache.capacity_gib must be positive when nvme_cache.enabled is true")
    if not isinstance(cfg.get("model"), str) or not cfg["model"]:
        raise ValueError("deployment model must be a non-empty string")
    release_name(cfg.get("release", ""))
    # Any rank count the site has nodes for. What a world actually supports is
    # not ours to assert: the engine rejects a geometry that does not divide by
    # it, and every rank's memory plan refuses a shape that does not fit. Both
    # failures name the model and the number; an allow-list here would only
    # forbid worlds that would have worked.
    if type(cfg.get("world_size")) is not int or cfg["world_size"] < 1:
        raise ValueError("deployment world_size must be a positive integer")
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict) or set(paths) - {"resident_cache"}:
        raise ValueError("deployment paths may only contain resident_cache; move site paths into .env")
    http = cfg.get("http", {})
    if not isinstance(http, dict) or set(http) - {"bind_host", "port", "max_body_bytes"}:
        raise ValueError("http may only contain bind_host, port and max_body_bytes")
    if "bind_host" in http:
        http_bind({"DGPP_HTTP_BIND": http["bind_host"]})
    if "port" in http and (type(http["port"]) is not int or not 1 <= http["port"] <= 65535):
        raise ValueError("http.port must be an integer in [1, 65535]")
    if "max_body_bytes" in http and (type(http["max_body_bytes"]) is not int
                                   or not 1 <= http["max_body_bytes"] <= (1 << 63) - 1):
        raise ValueError("http.max_body_bytes must be a positive 64-bit integer byte count")
    return cfg


def selected_nodes(values, world):
    nodes = site_nodes(values)
    if type(world) is not int or world < 1:
        raise ValueError("world size must be a positive integer")
    if world > len(nodes):
        raise ValueError(f"deployment needs {world} nodes, but DGPP_NODES contains only {len(nodes)}")
    return nodes[:world]


def release_name(value):
    if not isinstance(value, str) or (value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", value)):
        raise ValueError("release must be a version name without path or shell syntax")
    return value


def node_environments(values, nodes=None):
    """Resolve non-secret, per-node settings without expanding remote home paths."""
    nodes = site_nodes(values) if nodes is None else nodes
    common = {key: values[key] for key in NODE_KEYS if values.get(key)}
    if "HF_HUB_CACHE" not in common and values.get("HF_HOME"):
        common["HF_HUB_CACHE"] = values["HF_HOME"].rstrip("/") + "/hub"
    try:
        overrides = json.loads(values.get("DGPP_NODE_OVERRIDES") or "{}")
    except json.JSONDecodeError as error:
        raise ValueError("DGPP_NODE_OVERRIDES must be a JSON object") from error
    if not isinstance(overrides, dict) or set(overrides) - set(site_nodes(values)):
        raise ValueError("DGPP_NODE_OVERRIDES must map configured node names to settings")
    for host, override in overrides.items():
        if not isinstance(override, dict) or set(override) - set(NODE_KEYS):
            raise ValueError(f"invalid node settings for {host}; allowed: {', '.join(NODE_KEYS)}")
    result = []
    for host in nodes:
        env = {**common, **overrides.get(host, {})}
        if any(not isinstance(value, str) or any(c in value for c in "\0\r\n") for value in env.values()):
            raise ValueError(f"node settings for {host} must be single-line strings")
        devices = env.get("DGPP_ROCE_DEVICES", "").split()
        if len(devices) > 2:
            raise ValueError(f"at most two RoCE devices are supported for {host}")
        if len(set(devices)) != len(devices) or any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", d) for d in devices):
            raise ValueError(f"invalid or duplicate RoCE devices for {host}")
        gids = env.get("DGPP_ROCE_GID_INDICES", "").split()
        if gids and (len(gids) != len(devices) or any(not g.isdigit() or int(g) > 255 for g in gids)):
            raise ValueError(f"RoCE GID indices for {host} must match the device count and be in [0, 255]")
        for key in ("HF_HUB_CACHE", "DGPP_RESIDENT_CACHE_DIR"):
            if env.get(key) and not env[key].startswith(("/", "~/")):
                raise ValueError(f"{key} for {host} must be an absolute or ~/ path")
        result.append(env)
    return result


def http_bind(values):
    value = values.get("DGPP_HTTP_BIND", DEFAULTS["DGPP_HTTP_BIND"])
    if not isinstance(value, str):
        raise ValueError("HTTP bind host must be an IPv4 address string")
    try:
        ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as error:
        raise ValueError("DGPP_HTTP_BIND must be an IPv4 address (127.0.0.1, a local LAN address, or 0.0.0.0)") from error
    return value


def rank_environment(rank, values=None):
    values = settings() if values is None else values
    environments = node_environments(values)
    if not 0 <= rank < len(environments):
        raise ValueError("rank is outside DGPP_NODES")
    return environments[rank]


def cache_environment(values=None, rank=0):
    """Site cache defaults plus a node override, for local preparation tools."""
    values = settings() if values is None else values
    return {**values, **(rank_environment(rank, values) if values.get("DGPP_NODES") else {})}


def shell_prefix(env):
    assignments = []
    for key, value in sorted(env.items()):
        # Expand a remote user's home on the remote shell, not the head.
        quoted = '"$HOME"/' + shlex.quote(value[2:]) if value.startswith("~/") else shlex.quote(value)
        assignments.append(f"{key}={quoted}")
    return "env " + " ".join(assignments)


def resolve_config(path, values=None):
    values = settings() if values is None else values
    cfg = deployment(path)
    cfg["nodes"] = selected_nodes(values, cfg.pop("world_size"))
    cfg["ssh_user"] = ssh_user(values)
    cfg["ports"] = {name: port(values, name) for name in ("http", "fabric", "journal")}
    if cfg["ports"]["fabric"] == cfg["ports"]["journal"]:
        raise ValueError("DGPP_FABRIC_PORT and DGPP_JOURNAL_PORT must differ")
    cfg["paths"] = {**cfg.get("paths", {}), **site_paths(values)}
    cfg["http"] = {"bind_host": http_bind(values), "port": cfg["ports"]["http"], **cfg.get("http", {})}
    cfg["ports"]["http"] = cfg["http"]["port"]
    if len(cfg["nodes"]) > 1 and len(set(cfg["ports"].values())) != 3:
        raise ValueError("HTTP, fabric, and journal ports must differ for multi-node deployments")
    cfg["node_env"] = node_environments(values, cfg["nodes"])
    return cfg


def deployment_id(path):
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:16]


def run_paths(path, values=None, log_dir=None):
    values = settings() if values is None else values
    paths = site_paths(values)
    suffix = "/deployments/" + deployment_id(path)
    paths["stage_dir"] = paths["stage_dir"].rstrip("/") + suffix
    paths["log_dir"] = str(log_dir) if log_dir else paths["log_dir"].rstrip("/") + suffix
    return paths


def default_host():
    values = settings()
    path = config_path(values)
    bind = deployment(path).get("http", {}).get("bind_host", http_bind(values)) if Path(path).exists() else http_bind(values)
    return site_nodes(values)[0] if bind == "0.0.0.0" else bind


def http_port(values=None):
    values = settings() if values is None else values
    path = config_path(values)
    return deployment(path).get("http", {}).get("port", port(values, "http")) if Path(path).exists() else port(values, "http")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("shell", "config", "nodes", "head", "client-host", "peers", "user", "http-port", "resolve", "rank-prefix", "run-rank", "require-peers", "world", "model", "log-dir", "stage-dir"))
    parser.add_argument("--config", type=config_argument)
    parser.add_argument("--world", type=int)
    parser.add_argument("--rank", type=int, default=0)
    argv = sys.argv[1:]
    command = []
    if "--" in argv:
        at = argv.index("--")
        argv, command = argv[:at], argv[at + 1:]
    args = parser.parse_args(argv)
    values = settings()
    path = args.config or config_path(values)
    if args.command == "shell":
        site_paths(values)
        ssh_user(values)
        http_bind(values)
        ports = {name: port(values, name) for name in ("http", "fabric", "journal")}
        if ports["fabric"] == ports["journal"]:
            raise ValueError("DGPP_FABRIC_PORT and DGPP_JOURNAL_PORT must differ")
        if "DGPP_CLUSTER_CONFIG" in values:
            values["DGPP_CLUSTER_CONFIG"] = config_path(values)
        if "DGPP_ENV_FILE" in os.environ:
            values["DGPP_ENV_FILE"] = str(Path(os.environ["DGPP_ENV_FILE"]).expanduser().resolve())
        # NUL-delimited pairs let Bash assign literal values without eval.
        for key, value in values.items():
            sys.stdout.buffer.write(key.encode() + b"\0" + value.encode() + b"\0")
    elif args.command == "config":
        print(path)
    elif args.command == "head":
        print(site_nodes(values)[0])
    elif args.command == "client-host":
        print(default_host())
    elif args.command in ("log-dir", "stage-dir"):
        print(run_paths(path, values, os.environ.get("DGPP_SERVE_LOG"))[args.command.replace("-", "_")])
    elif args.command == "model":
        print(deployment(path)["model"])
    elif args.command == "rank-prefix":
        print(shell_prefix(rank_environment(args.rank, values)))
    elif args.command == "run-rank":
        if not command:
            parser.error("run-rank needs -- COMMAND [ARGS...]")
        env = dict(os.environ)
        env.update({key: os.path.expanduser(value) for key, value in rank_environment(args.rank, values).items()})
        os.execvpe(command[0], command, env)
    elif args.command == "require-peers":
        actual = deployment(path)["world_size"]
        if actual < 2:
            raise ValueError(f"this procedure requires a world with peers; selected deployment has world_size={actual}")
    elif args.command == "world":
        print(deployment(path)["world_size"])
    elif args.command == "user":
        print(ssh_user(values))
    elif args.command == "http-port":
        print(http_port(values))
    elif args.command == "resolve":
        print(json.dumps(resolve_config(path, values), indent=2))
    else:
        world = args.world if args.world is not None else deployment(path)["world_size"]
        nodes = selected_nodes(values, world)
        print(" ".join(nodes[1:] if args.command == "peers" else nodes))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"site configuration: {error}", file=sys.stderr)
        sys.exit(2)
