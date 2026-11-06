"""Guide a source checkout from site configuration to a serving preflight.

Uses only the standard library. Run setup.sh on rank 0; --help lists the
unattended, read-only and offline checkpoint options.
"""
import sys

# Before anything else is imported: an older interpreter must get this message,
# not a traceback from a project module (so this file keeps to syntax every
# Python 3 still in the field parses; bootstrap_test pins that).
if sys.version_info < (3, 10):
    print("DGPP setup needs Python 3.10+; select a newer python3 in PATH.", file=sys.stderr)
    raise SystemExit(2)

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import tempfile

import cluster_doctor
import discover_roce
import site_env

ROOT = Path(__file__).resolve().parents[1]
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=yes"]
RUNTIME_PACKAGES = ["python3", "rdma-core", "libibverbs1", "ibverbs-providers", "libnl-3-200",
                    "libnl-route-3-200", "libstdc++6", "libssl3t64", "libpcre2-8-0", "openssh-client",
                    "openssh-server", "rsync", "curl", "jq", "iproute2", "coreutils"]
BUILD_PACKAGES = ["git", "build-essential", "cmake", "pkg-config", "python3-venv",
                  "libibverbs-dev", "libssl-dev", "libpcre2-dev", "poppler-utils"]


def ask(label, default=""):
    try:
        reply = input(f"{label}" + (f" [{default}]" if default else "") + ": ").strip()
    except EOFError as error:
        raise ValueError("input ended; rerun interactively or supply --non-interactive options") from error
    return reply or default


def yes(label):
    return ask(label + " (y/N)", "n").lower() in ("y", "yes")


def run(command, **kwargs):
    command = [str(part) for part in command]
    print("+ " + shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, **kwargs)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; fix the error above and rerun setup")
    return result


def absolute(path):
    return Path(path).expanduser().resolve()


def site_values(path):
    return {**site_env.DEFAULTS, **(site_env.read_env(path) if path.exists() else {}),
            **{key: os.environ[key] for key in site_env.SITE_KEYS if key in os.environ}}


def change(values, updates, key, value):
    exported = os.environ.get(key)
    if exported is not None and key == "DGPP_CLUSTER_CONFIG":
        exported = site_env.config_path({key: exported})
    if exported is not None and key == "DGPP_NODES":
        exported = " ".join(exported.split())
    if exported is not None and exported != value:
        raise ValueError(f"exported {key} overrides the site file; unset it before changing this setting")
    updates[key] = value
    values[key] = value


def env_literal(value):
    if any(c in value for c in "\r\n\0"):
        raise ValueError("site settings must be single-line literals")
    for quote in ('"', "'"):
        if quote not in value:
            return quote + value + quote
    raise ValueError("a site setting contains both quote styles; choose a path without quotes")


def save_env(path, updates):
    """Replace only requested settings, retaining credentials and comments verbatim."""
    if not updates and path.exists():
        return
    old = path.read_text() if path.exists() else "# DGPP site settings; parsed as data, never sourced.\n"
    if path.exists():
        site_env.read_env(path)  # Refuse ambiguous/duplicate settings before writing.
    pending = dict(updates)
    lines = []
    for line in old.splitlines(keepends=True):
        match = re.match(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=", line)
        if match and match[1] in pending:
            key = match[1]
            line = f"{key}={env_literal(pending.pop(key))}\n"
        lines.append(line)
    content = "".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    content += "".join(f"{key}={env_literal(value)}\n" for key, value in pending.items())
    if content == old:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(content)
            output.flush()
            site_env.read_env(temporary)
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def templates():
    return [(path, site_env.deployment(path)) for path in sorted((ROOT / "deploy").glob("*.example.json"))]


def show_templates(items):
    for number, (path, cfg) in enumerate(items, 1):
        print(f"  {number}. {cfg['model']} — {cfg['world_size']} node(s)\n     {path.name}")


def choose_config(args, values, interactive):
    source = None
    if args.template:
        source = Path(args.template).expanduser()
        if not source.exists():
            source = ROOT / "deploy" / args.template
        if not source.name.endswith(".example.json") or not source.is_file():
            raise ValueError("--template must name a deploy/*.example.json file; use --list-templates")
        source = source.resolve()
    if args.config:
        target = absolute(args.config)
    elif source:
        target = source.with_name(source.name.replace(".example.json", ".json"))
    elif values.get("DGPP_CLUSTER_CONFIG"):
        target = Path(site_env.config_path(values))
    elif args.check:
        target = Path(site_env.config_path(values))
    elif interactive:
        print("Supported deployments use one GB10 Spark per rank. Choose the nodes you intend to use.")
        count = ask("Number of Sparks (1, 2 or 4)", str(len((args.nodes or values.get("DGPP_NODES", "")).split()) or 1))
        if count not in ("1", "2", "4"):
            raise ValueError("choose 1, 2 or 4 nodes from the supported deployment catalogue")
        choices = [(p, c) for p, c in templates() if c["world_size"] == int(count)]
        show_templates(choices)
        choice = ask("Deployment number", "1")
        if not choice.isdigit() or not 1 <= int(choice) <= len(choices):
            raise ValueError("invalid deployment number")
        source = choices[int(choice) - 1][0]
        target = source.with_name(source.name.replace(".example.json", ".json"))
    else:
        raise ValueError("select --template FILE or an existing --config FILE; use --list-templates")
    if target.name.endswith(".example.json"):
        raise ValueError("--config must name a local deployment copy, not a tracked .example.json template")
    if target.exists():
        cfg = site_env.deployment(target)
        if source:
            template = site_env.deployment(source)
            if (cfg["model"], cfg["world_size"]) != (template["model"], template["world_size"]):
                raise ValueError(f"{target} already selects a different deployment; choose another --config path")
        source = target  # Preserve all locally tuned engine settings.
    elif source:
        cfg = site_env.deployment(source)
    else:
        raise ValueError(f"no deployment at {target}; select a --template to create it")
    if cfg.get("release"):
        raise ValueError("setup builds from source; use a deployment without 'release', or follow docs/release-install.md")
    return source, target, cfg


def configure(args, values, interactive):
    source, target, deployment = choose_config(args, values, interactive)
    updates = {}
    print(f"Deployment: {deployment['model']} on {deployment['world_size']} node(s)\nConfig: {target}")
    world = deployment["world_size"]
    nodes = args.nodes
    if nodes is None and interactive:
        default = values.get("DGPP_NODES", "") or ("127.0.0.1" if world == 1 else "")
        print("Put this machine (rank 0) first. Management IPs, fabric IPs, or a mixture are supported.")
        if world > 1:
            print("Rank 0 must reach each peer over SSH. Every peer must reach the selected rank-0 address for TCP coordination.")
            print("If only rank 0 has a separate management IP, you can use fabric IPs for all entries, including rank 0.")
            print("Choose a rank-0 address reachable from the peers; your operator login may use its other management IP.")
            print("You'll select RoCE devices/GIDs afterward; the same interfaces can carry SSH/control traffic and RDMA.")
        nodes = ask("SSH/control addresses in rank order, separated by spaces", default)
    if nodes is None and not values.get("DGPP_NODES") and world == 1:
        nodes = "127.0.0.1"
    if nodes is not None:
        change(values, updates, "DGPP_NODES", " ".join(nodes.split()))
    fields = [("HF_HUB_CACHE", args.cache_dir, "Checkpoint cache on each node (~/ expands per node)",
               values.get("HF_HUB_CACHE") or (values.get("HF_HOME") or "~/.cache/huggingface").rstrip("/") + "/hub"),
              ("DGPP_RESIDENT_CACHE_DIR", args.resident_cache_dir, "Resident image cache on each node",
               values.get("DGPP_RESIDENT_CACHE_DIR") or deployment.get("paths", {}).get("resident_cache") or "~/.cache/dgpp/resident")]
    if world > 1 or args.ssh_user is not None:
        fields.insert(0, ("DGPP_SSH_USER", args.ssh_user, "SSH login on peers", site_env.ssh_user(values)))
    if "bind_host" not in deployment.get("http", {}):
        fields.append(("DGPP_HTTP_BIND", args.http_bind,
                       "HTTP bind address (0.0.0.0 = all interfaces, 127.0.0.1 = local only)",
                       values["DGPP_HTTP_BIND"]))
    else:
        print(f"HTTP bind address: {deployment['http']['bind_host']} (set by deployment http.bind_host)")
    if "port" not in deployment.get("http", {}):
        fields.append(("DGPP_HTTP_PORT", args.http_port, "HTTP port on rank 0", values["DGPP_HTTP_PORT"]))
    for key, option, label, default in fields:
        if option is not None or interactive:
            value = str(option) if option is not None else ask(label, default)
            change(values, updates, key, value)
    if args.http_port is not None and "port" in deployment.get("http", {}) and args.http_port != deployment["http"]["port"]:
        raise ValueError("this deployment sets http.port; edit it there instead of using --http-port")
    if args.http_bind is not None and "bind_host" in deployment.get("http", {}) and args.http_bind != deployment["http"]["bind_host"]:
        raise ValueError("this deployment sets http.bind_host; edit it there instead of using --http-bind")
    # Keep deployment selection persistent without relying on a shell variable.
    change(values, updates, "DGPP_CLUSTER_CONFIG", str(target))
    cfg = site_env.resolve_config(source, values)
    if len(cfg["nodes"]) > 1 and any(node in ("localhost", "127.0.0.1", "::1") for node in cfg["nodes"]):
        raise ValueError("multi-node deployments need network addresses, not loopback addresses")
    print(f"HTTP: {cfg['http']['bind_host']}:{cfg['http']['port']} (the service has no authentication or TLS)")
    print("Each node needs the full checkpoint plus resident-cache and transfer space.")
    if world == 4 and "GLM-5.3-Flash-NVFP4-FP8" in cfg["model"]:
        print("For the Flash hybrid, plan roughly 250 GiB per node; other models need their own storage budget.")
    print("Existing per-node cache and RoCE overrides are retained; free space is reported below.")
    return source, target, cfg, updates


def check_ssh(cfg):
    failures = []
    for host in cfg["nodes"][1:]:
        target = f"{cfg['ssh_user']}@{host}"
        try:
            # No stdin: a batch-mode reachability probe must not consume the
            # operator's type-ahead (or a piped answer file) meant for the prompts.
            result = subprocess.run(["ssh", *SSH_OPTIONS, target, "true"], stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=15)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "SSH command failed")
            print(f"OK SSH {target}")
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            print(f"FAIL SSH {target}: {error}")
            print(f"  Verify the host fingerprint, then connect with: ssh {shlex.quote(target)}")
            print(f"  Install your public key with ssh-copy-id {shlex.quote(target)}; load passphrase keys with ssh-add.")
            failures.append(host)
    if failures:
        raise ValueError("fix rank-0-to-peer SSH access and rerun; see docs/getting-started.md#3-set-your-node-addresses-and-ssh-user")


def configure_roce(cfg, values, interactive):
    if len(cfg["nodes"]) == 1:
        print("One node: no RoCE lane configuration is needed (libibverbs is still required).")
        return {}
    report = discover_roce.inspect_cluster(cfg)
    discover_roce.print_report(report, cfg["nodes"])
    if report["errors"]:
        raise ValueError("RoCE discovery failed; fix the reported peer/driver errors and rerun")
    if not interactive:
        return {}
    overrides = json.loads(values.get("DGPP_NODE_OVERRIDES") or "{}")
    print("RoCE devices may be the same interfaces used by your SSH/control addresses; a separate management network is optional.")
    print("Select corresponding subnets in the same lane order on every node. Software cannot verify your cabling.")
    for rank, host in enumerate(cfg["nodes"]):
        rows = {row["device"]: row for row in report["inventories"][host] if row["usable"]}
        current = cfg["node_env"][rank]
        default = current.get("DGPP_ROCE_DEVICES", "") or " ".join(sorted(rows))
        devices = ask(f"{host}: one or two verbs devices in lane order", default).split()
        if not 1 <= len(devices) <= 2 or len(set(devices)) != len(devices) or any(d not in rows for d in devices):
            raise ValueError(f"{host}: select one or two distinct locally eligible port-1 devices from the inventory")
        old_devices = current.get("DGPP_ROCE_DEVICES", "").split()
        old_indices = dict(zip(old_devices, current.get("DGPP_ROCE_GID_INDICES", "").split()))
        indices = []
        for device in devices:
            candidates = [str(gid["index"]) for gid in rows[device]["gids"]]
            default = old_indices.get(device, candidates[0] if len(candidates) == 1 else "")
            index = ask(f"{host} {device}: GID index ({', '.join(candidates)})", default)
            if index not in candidates:
                raise ValueError(f"{host} {device}: choose a routable RoCE-v2 GID from the inventory")
            indices.append(index)
        overrides[host] = {**overrides.get(host, {}), "DGPP_ROCE_DEVICES": " ".join(devices),
                           "DGPP_ROCE_GID_INDICES": " ".join(indices)}
    updates = {}
    change(values, updates, "DGPP_NODE_OVERRIDES", json.dumps(overrides, separators=(",", ":")))
    site_env.node_environments(values, cfg["nodes"])
    return updates


def cuda_compiler():
    if os.environ.get("CUDACXX"):
        return shlex.split(os.environ["CUDACXX"])
    if os.environ.get("CUDAToolkit_ROOT"):
        return [str(Path(os.environ["CUDAToolkit_ROOT"]) / "bin/nvcc")]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return [nvcc]
    for root in (os.environ.get("CUDA_PATH", ""), "/usr/local/cuda"):
        if root and (Path(root) / "bin/nvcc").is_file():
            return [str(Path(root) / "bin/nvcc")]
    return ["nvcc"]


def build_checks(needs_download, *, skip_build=False):
    failures = 0
    def check(label, command, valid, hint, **kwargs):
        nonlocal failures
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, **kwargs)
            ok = result.returncode == 0 and valid(result.stdout + result.stderr)
            detail = (result.stdout + result.stderr).strip()
        except (OSError, subprocess.SubprocessError) as error:
            ok, detail = False, str(error)
        print(f"{'OK' if ok else 'FAIL'} {label}: " + (detail.splitlines()[0] if ok and detail else ("available" if ok else hint)))
        if not ok:
            failures += 1
    def version(text, pattern, minimum):
        match = re.search(pattern, text)
        return bool(match and tuple(map(int, match.groups())) >= minimum)
    if needs_download:
        check("Python venv/ensurepip", [sys.executable, "-c", "import venv, ensurepip"], lambda s: True,
              "install python3-venv (or the venv package matching your Python version)")
    if skip_build:
        return failures
    check("CMake 3.25+", ["cmake", "--version"], lambda s: version(s, r"cmake version (\d+)\.(\d+)", (3, 25)),
          "install CMake 3.25 or newer")
    generator = "ninja" if "Ninja" in os.environ.get("CMAKE_GENERATOR", "") else "make"
    check(generator, [generator, "--version"], lambda s: True, f"install {generator}")
    check("Git", ["git", "--version"], lambda s: True, "install git")
    cxx = shlex.split(os.environ.get("CXX") or "c++")
    check("C++20 compiler", [*cxx, "-std=c++20", "-x", "c++", "-fsyntax-only", "-"], lambda s: True,
          "install a C++20 compiler (tested: GCC 13 / build-essential)", input="#include <span>\nint main() { std::span<int> s; }\n")
    check("libibverbs headers", [*cxx, "-x", "c++", "-fsyntax-only", "-"], lambda s: True,
          "install libibverbs-dev (also required for one node)", input="#include <infiniband/verbs.h>\n")
    check("CUDA 13+ compiler", [*cuda_compiler(), "--version"],
          lambda s: version(s, r"release (\d+)\.(\d+)", (13, 0)),
          "install the DGX OS CUDA 13 toolkit or set CUDACXX=/path/to/cuda/bin/nvcc; nvidia-smi alone is not a toolkit check")
    print("CMake will check CUDA/cuBLASLt development libraries and PCRE2; without system PCRE2 it fetches pinned source.")
    return failures


def package_help():
    print("Ubuntu/DGX OS packages on every node:\n  sudo apt-get update\n  " +
          shlex.join(["sudo", "apt-get", "install", "-y", *RUNTIME_PACKAGES]))
    print("Additional rank-0 packages:\n  " + shlex.join(["sudo", "apt-get", "install", "-y", *BUILD_PACKAGES]))
    print("Use --install-system-deps to run these on all selected nodes. Other distributions need equivalent packages.")
    print("Provision NVIDIA drivers and CUDA 13 through DGX OS; peers need compatible cudart/cuBLASLt runtimes.")


def install_packages(cfg, *, interactive=False):
    # System changes require an explicit flag or the interactive yes prompt.
    sudo = ["sudo"] if interactive else ["sudo", "-n"]
    local_sudo = sudo if os.geteuid() else []
    if not shutil.which("apt-get"):
        raise ValueError("automatic system-package installation requires apt-get; install equivalent packages manually")
    run([*local_sudo, "apt-get", "update"])
    install = ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y",
               "-o", "Dpkg::Options::=--force-confold"]
    run([*local_sudo, *install, *RUNTIME_PACKAGES, *BUILD_PACKAGES])
    check_ssh(cfg)
    for host in cfg["nodes"][1:]:
        command = shlex.join([*sudo, "apt-get", "update"]) + " && " + shlex.join([*sudo, *install, *RUNTIME_PACKAGES])
        run(["ssh", *(["-t"] if interactive else []), *SSH_OPTIONS, f"{cfg['ssh_user']}@{host}", command])


def model_action(requested, cfg):
    if requested != "auto":
        return requested
    try:
        root = cluster_doctor.cache_root({**os.environ, **cfg["node_env"][0]})
        cluster_doctor.checkpoint_size(cluster_doctor.cached_snapshot(cfg["model"], root))
        table_model = (cfg.get("engine", {}) or {}).get("ngram_table_model")
        if table_model:
            cluster_doctor.shard_bytes(cluster_doctor.cached_snapshot(table_model, root))
        return "sync"
    except (OSError, ValueError, KeyError, TypeError):
        return "download"


def prepare_model(action, config):
    python = sys.executable
    if action == "download":
        venv = ROOT / ".venv"
        if not (venv / "pyvenv.cfg").exists():
            if venv.exists():
                raise ValueError(".venv exists but is not a virtual environment; move it aside or repair it before rerunning")
            run([python, "-m", "venv", venv])
        python = str(venv / "bin/python")
        run([python, "-m", "pip", "install", "-r", ROOT / "requirements-download.txt"])
        print("Gated/private models need rank-0 HF authentication: .venv/bin/hf auth login (or export HF_TOKEN).", flush=True)
    command = [python, ROOT / "scripts/download_model.py", "--config", config]
    if action in ("sync", "verify"):
        command.append("--sync-only" if action == "sync" else "--verify-only")
    run(command)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--list-templates", action="store_true", help="list supported model/node combinations and exit")
    result.add_argument("--template", help="deploy/*.example.json filename or path; existing local copies are preserved")
    result.add_argument("--config", type=site_env.config_argument, help="existing deployment, or output path when selecting a template")
    result.add_argument("--env-file", help="site file (default: DGPP_ENV_FILE or repository .env)")
    result.add_argument("--nodes", help="quoted SSH/control addresses, rank 0 first; management IPs, fabric IPs or a mixture")
    result.add_argument("--ssh-user", help="peer login")
    result.add_argument("--cache-dir", help="HF cache on each node, absolute or ~/ path; existing per-node overrides win")
    result.add_argument("--resident-cache-dir", help="resident image cache on each node; existing per-node overrides win")
    result.add_argument("--http-port", type=int, help="HTTP port unless the deployment sets http.port")
    result.add_argument("--http-bind", help="HTTP IPv4 bind address (default: 0.0.0.0, all interfaces); deployment http.bind_host takes precedence")
    result.add_argument("--non-interactive", action="store_true", help="use flags/existing settings; never prompt")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only prerequisite checks; no files, installation, build or download")
    mode.add_argument("--configure-only", action="store_true", help="save deployment/site settings and exit before SSH or preparation")
    result.add_argument("--install-system-deps", action="store_true", help="install Ubuntu/Debian packages on all nodes using sudo; excludes driver/CUDA")
    result.add_argument("--model-action", choices=("auto", "download", "sync", "verify"), default="auto",
                        help="auto reuses a complete head cache, otherwise downloads; sync/verify never contact the Hub")
    result.add_argument("--skip-build", action="store_true", help="reuse the existing server; final preflight still checks it")
    result.add_argument("--jobs", type=int, default=4, help="parallel build jobs (default: 4)")
    result.add_argument("--cmake-arg", action="append", default=[], help="extra configure argument, e.g. --cmake-arg=-DFETCHCONTENT_SOURCE_DIR_PCRE2=/path")
    result.add_argument("--start", action="store_true", help="start after successful full preflight; never replace a running deployment")
    return result


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.list_templates:
        show_templates(templates())
        return 0
    if args.jobs < 1:
        ap.error("--jobs must be positive")
    if (args.check or args.configure_only) and (args.install_system_deps or args.start):
        ap.error("--check/--configure-only cannot install packages or start services")
    interactive = sys.stdin.isatty() and not args.non_interactive and not args.check
    env_file = absolute(args.env_file or os.environ.get("DGPP_ENV_FILE", ROOT / ".env"))
    values = site_values(env_file)
    source, config, cfg, updates = configure(args, values, interactive)
    if not args.check:
        # Validate before writes and refuse to overwrite a local deployment.
        if not config.exists():
            config.parent.mkdir(parents=True, exist_ok=True)
            with config.open("x") as output:
                output.write(source.read_text())
        save_env(env_file, updates)
        os.environ["DGPP_ENV_FILE"] = str(env_file)
        print(f"Saved site settings: {env_file}")
    if args.configure_only:
        print("Configuration saved. Dependencies, networking, build and checkpoint readiness have not been checked.")
        return 0
    cluster_doctor.require_head(cfg["nodes"][0])
    if args.install_system_deps or interactive:
        package_help()
        if interactive and not args.install_system_deps:
            args.install_system_deps = yes("Install the listed system packages on all selected nodes")
    if args.install_system_deps:
        install_packages(cfg, interactive=interactive)
    else:
        check_ssh(cfg)
    paths = site_env.run_paths(config, values)
    action = model_action(args.model_action, cfg)
    if interactive:
        print(f"Checkpoint action: {action}. Downloads and peer transfers can be large; stop deployments using this checkpoint first.")
        action = ask("Checkpoint action (download, sync or verify)", action)
        if action not in ("download", "sync", "verify"):
            raise ValueError("choose download, sync or verify")
    def prerequisites():
        failures = build_checks(action == "download", skip_build=args.skip_build)
        failures += cluster_doctor.check_cluster(cfg, None, paths["log_dir"], paths["stage_dir"], cfg["ssh_user"], preparing=True)
        return failures
    # Lane choices come before preflight: automatic selection may include more
    # than two devices, or devices on an unintended fabric.
    if not args.check:
        changes = configure_roce(cfg, values, interactive)
        save_env(env_file, changes)
        cfg = site_env.resolve_config(config, values)
    failed = prerequisites()
    if failed:
        package_help()
        raise ValueError("prerequisites failed; fix the reported checks and rerun setup")
    if args.check:
        print("Prerequisites passed. No files changed; server build and checkpoint readiness have not been checked.")
        return 0
    build_dir = Path(values.get("DGPP_BUILD_DIR") or ROOT / "build-release").expanduser()
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    if not args.skip_build:
        run(["cmake", "--preset", "release", "-B", build_dir, "-DDGPP_ENABLE_IBV=ON", *args.cmake_arg])
        run(["cmake", "--build", build_dir, "--target", "dgpp_serve_app", "-j", args.jobs])
    prepare_model(action, config)
    launcher = [sys.executable, ROOT / "scripts/dgpp-cluster"]
    run([*launcher, "doctor", "--config", config])
    print("\nSetup complete: build, checkpoint and serving preflight passed.")
    print("Preflight does not test physical cabling or end-to-end RDMA; startup checks the model memory plan.")
    prefix = ["env", f"DGPP_ENV_FILE={env_file}", "python3", str(ROOT / "scripts/dgpp-cluster")]
    for verb in ("up", "status", "down"):
        print(shlex.join([*prefix, verb, "--config", str(config)]))
    host = cfg["http"]["bind_host"]
    print(f"After READY: curl --fail http://{'127.0.0.1' if host == '0.0.0.0' else host}:{cfg['http']['port']}/v1/models")
    if args.start:
        run([*launcher, "up", "--config", config])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nSetup interrupted. Rerun to reuse saved settings and completed work.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"DGPP setup: {error}", file=sys.stderr)
        raise SystemExit(2)
