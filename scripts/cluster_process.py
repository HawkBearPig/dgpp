"""Start and signal only processes recorded for one deployment on this host.

Records include the Linux boot ID, process start time and owner. A stale PID
file never authorizes a signal to a reused PID. Launches are serialized with
flock; each process gets its own session for wrapper/child cleanup.
DGPP_NO_SWAP=1 starts the command in a systemd user scope whose swap limit
is zero. OS swap and other processes are unaffected; scope creation must
succeed before the command runs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def identity(pid):
    try:
        proc = Path("/proc") / str(pid)
        fields = (proc / "stat").read_text().rpartition(") ")[2].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start": fields[19], "uid": proc.stat().st_uid,
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def read_record(path):
    try:
        record = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    if not isinstance(record, dict) or type(record.get("pid")) is not int or record["pid"] <= 1:
        raise ValueError(f"invalid process record: {path}; inspect it before removing it")
    if set(record) != {"pid", "start", "uid", "boot"}:
        raise ValueError(f"incomplete process record: {path}")
    return record


def running(path):
    record = read_record(path)
    return record if record and record["uid"] == os.getuid() and identity(record["pid"]) == record else None


def send_signal(path, number):
    record = running(path)
    if not record:
        return False
    # The process must still be the session leader created by launch().
    try:
        if os.getsid(record["pid"]) != record["pid"]:
            raise ValueError("recorded process is not its own session leader; refusing group signal")
        if identity(record["pid"]) != record:
            return False
        os.killpg(record["pid"], number)
        return True
    except ProcessLookupError:
        return False


def launch(state, command, log, cwd, env=None, pass_fds=()):
    no_swap = (os.environ if env is None else env).get("DGPP_NO_SWAP", "0")
    if no_swap not in ("0", "1"):
        raise ValueError("DGPP_NO_SWAP must be 0 or 1")
    if no_swap == "1":
        if pass_fds:
            raise ValueError("inherited control descriptors require a direct process launch (DGPP_NO_SWAP=0)")
        command = ["systemd-run", "--user", "--scope", "--quiet",
                   "--property=MemorySwapMax=0", "--", *command]
    state = Path(state)
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.with_suffix(state.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if running(state):
            raise ValueError(f"deployment is already running ({state}); stop it before replacing it")
        with open(log, "ab") as output:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, pass_fds=pass_fds)
        try:
            record = identity(proc.pid)
            if record is None:
                raise RuntimeError(f"process exited before its identity could be recorded; see {log}")
            with tempfile.NamedTemporaryFile(mode="w", dir=state.parent, delete=False) as temp:
                json.dump(record, temp)
                temporary_path = Path(temp.name)
            try:
                os.replace(temporary_path, state)
            finally:
                temporary_path.unlink(missing_ok=True)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        return proc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "run", "status", "signal"))
    parser.add_argument("--state", required=True)
    parser.add_argument("--log")
    parser.add_argument("--cwd")
    parser.add_argument("--signal", choices=("INT", "TERM", "KILL"), default="INT")
    argv = sys.argv[1:]
    at = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:at])
    if args.action in ("launch", "run"):
        command = argv[at + 1:]
        if not command or not args.log or not args.cwd:
            parser.error("launch needs --log, --cwd, and -- COMMAND [ARGS...]")
        proc = launch(args.state, command, args.log, args.cwd)
        if args.action == "run":
            return proc.wait()
        print(proc.pid)
    elif args.action == "status":
        record = running(args.state)
        print(json.dumps(record))
        return 0 if record else 1
    else:
        send_signal(args.state, getattr(signal, "SIG" + args.signal))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"process control: {error}", file=sys.stderr)
        raise SystemExit(2)
