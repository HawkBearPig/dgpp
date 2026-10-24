#!/usr/bin/env python3
"""Every tracked benchmark for one deployment template, in one command.

The rows docs/benchmarks.md keeps for a deployment all come from one boot of
the template as shipped (plus one boot without its drafter, and one per
extra decode mode asked for). This runner boots it on the bench port, runs
each measurement with the maintained clients, checks the run's validity,
tears the world down, and writes a record directory the `summarize` and
`rows` commands read back. Records stay out of Git (benchmarks/results is
ignored); the doc carries the cells.

  serve_campaign.py run --config deploy/cluster_<model>_<quant>_w<n>.example.json --out benchmarks/results/<label>
                        [--bin build-release/dgpp-serve] [--port 18081]
                        [--phases serving,prefill,context,quality,api,transcripts,plain]
                        [--mode NAME="<knobs>" ...] [--no-swap-scope] [--dry-run]
  serve_campaign.py summarize DIR                     every cell, labelled
  serve_campaign.py rows DIR --label "Qwen3.8-27B FP8" --nodes 1 --options "DFlash2 drafter, ..."
                                                      the doc's table rows for this deployment

What one run measures (docs/benchmarks.md "Workload and timing" is the
definition of record):
  serving      timed_load at C1/C2/C4/C8 over the five prompt classes, three
               repetitions, 256 tokens: the engine's decode-only rate per class
               and concurrency (the headline), the wall rate (prefill and
               queueing included) beside it.
  prefill      serve_prefill_probe at 512 / 2K / 8K / 32K, three distinct cold
               prefixes each, thinking off: cold prefill seconds.
  context      serve_context_matrix at 0 / 32K / 64K / 128K / 256K / 512K: cold
               prefill per bucket, decode ms a pass and tok/s from three requests.
  quality      serve_eval on the 300-item HumanEval / GSM8K / extraction set at
               the deployment's slot count; responses cut at the standard cap
               are asked once more at the 8,192 retry cap (recorded, not scored).
  api          serve_api_check: stop, n, logit_bias, usage.
  transcripts  serve_load C1, five classes x 3 greedy transcripts in the shipped
               mode, and the isolation check (prompt 0 alone, then beside
               slots-1 others).
  plain        a second boot of the template without its drafting keys (mtp
               false): the same transcripts, for the plain/default match.
  --mode       a boot per extra decode mode (knobs appended to every rank): the
               C1 serving sweep, for the decode-mode table.
Validity, recorded with the cells: the source revision and the binary's hash,
the resolved deployment, every rank's VmSwap (must stay 0) and binary hash
before and after, zero cached tokens in every cold probe, the ranks' operation
streams identical at teardown. Ranks run in the launcher's swap-limited scopes
(DGPP_NO_SWAP=1) unless --no-swap-scope.

"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLASSES = ("prose", "code", "json", "math", "chat")
BUCKETS = (0, 32768, 65536, 131072, 262144, 524288)
PREFILL_TARGETS = (512, 2048, 8192, 32768)
PHASES = ("serving", "prefill", "context", "quality", "api", "transcripts", "plain")


# ---- running ---------------------------------------------------------------------------------------------

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args, path, env, timeout=3600, check=True):
    """Run one command with its stdout+stderr in `path` and a receipt beside it."""
    receipt = {"command": [str(a) for a in args], "started": utcnow()}
    print(receipt["started"][11:19], path.name, flush=True)
    with path.open("w") as f:
        try:
            r = subprocess.run(receipt["command"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            receipt["returncode"] = r.returncode
        except subprocess.TimeoutExpired:
            receipt["returncode"] = 124
    receipt["finished"] = utcnow()
    path.with_suffix(path.suffix + ".command.json").write_text(json.dumps(receipt, indent=2))
    if check and receipt["returncode"]:
        raise RuntimeError(f"{path.name} failed: {receipt['returncode']}")
    return receipt["returncode"]


def scoped_env(no_swap_scope):
    """The launcher's environment: the swap-limited scopes (DGPP_NO_SWAP=1 plus the user bus a non-login shell lacks)."""
    env = dict(os.environ)
    if no_swap_scope:
        env.pop("DGPP_NO_SWAP", None)
    else:
        env["DGPP_NO_SWAP"] = "1"
        uid = os.getuid()
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={env['XDG_RUNTIME_DIR']}/bus")
    return env


def deployment_from(template, port):
    cfg = json.loads(pathlib.Path(template).read_text())
    cfg["http"] = {"bind_host": "127.0.0.1", "port": int(port)}
    return cfg


DRAFT_KEYS = ("dflash_model", "draft_head", "dflash_weights", "mtp_schedule", "mtp_schedule_row_ms", "mtp_schedule_base_ms",
              "mtp_schedule_lambda", "mtp_schedule_min_depth", "mtp_schedule_adapt", "mtp_schedule_sampled_scale", "mtp_draft",
              "mtp_draft_temperature", "mtp_verify", "mtp_depth", "lookup_draft", "lookup_tail", "dflash_batch_rows", "dflash_depth")


def drafts(engine):
    """Whether this template drafts at all (a block drafter or the native MTP layer)."""
    return bool(engine.get("dflash_model") or engine.get("mtp"))


def plain_deployment(cfg):
    """The same deployment with its drafting switched off: the draft keys removed and mtp false, so the plain-decode
    boot is the template's own engine settings otherwise (a knob such as --no-dflash would leave the schedule keys
    behind, which the server refuses without a drafter)."""
    plain = json.loads(json.dumps(cfg))
    for k in DRAFT_KEYS:
        plain["engine"].pop(k, None)
    plain["engine"]["mtp"] = False
    return plain


PROBE = ("import hashlib,json,pathlib,sys;st=json.loads(pathlib.Path(sys.argv[1]).read_text());p=pathlib.Path('/proc')/str(st['pid']);"
         "s=dict(l.split(':',1) for l in (p/'status').read_text().splitlines() if ':' in l);"
         "print(json.dumps(dict(pid=st['pid'],sha=hashlib.sha256((p/'exe').read_bytes()).hexdigest(),"
         "swap_kib=int(s['VmSwap'].split()[0]),rss_kib=int(s['VmRSS'].split()[0]))))")


class Boot:
    """One world: up on the bench port, the checks, down with the op-stream match."""

    def __init__(self, d, cfg, env, binary, knobs=""):
        self.d, self.cfg, self.env, self.binary, self.knobs = d, cfg, env, binary, knobs
        self.world = d / "world"
        self.config = d / "deployment.json"
        self.port = str(cfg["http"]["port"])
        self.host = "127.0.0.1"
        self.world_size = int(cfg.get("world_size") or 1)
        self.status = {}
        self.sha = hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest()

    def launcher(self, verb, extra):
        return ["python3", "scripts/dgpp-cluster", verb, "--config", str(self.config), "--bin", self.binary, "--log-dir", str(self.world), *extra]

    def up(self):
        self.d.mkdir(parents=True, exist_ok=False)
        self.config.write_text(json.dumps(self.cfg, indent=2))
        (self.d / "binary.sha256").write_text(self.sha + "\n")
        (self.d / "source.txt").write_text(
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout)
        extra = [f"--knobs={self.knobs}"] if self.knobs else []  # one token: a knob that starts with -- is otherwise read as the next option
        run(self.launcher("up", extra), self.d / "up.log", self.env, timeout=1800)
        resolved = json.loads((self.world / "cluster.resolved.json").read_text())
        self.world_size = len(resolved["nodes"])
        self.snapshot("before")
        run(["python3", "scripts/serve_bench.py", self.host, self.port, "32", "warm"], self.d / "warm.log", self.env)

    def snapshot(self, label):
        """Every rank's dgpp-serve: VmSwap (must be 0) and the binary's hash (must be the one under test)."""
        cfg = json.loads((self.world / "cluster.resolved.json").read_text())
        out = []
        for rank, host in enumerate(cfg["nodes"]):
            state = str(self.world / "rank0.process.json") if rank == 0 else cfg["paths"]["stage_dir"] + f"/rank{rank}.process.json"
            cmd = ["python3", "-c", PROBE, state] if rank == 0 else \
                ["ssh", "-o", "BatchMode=yes", cfg["ssh_user"] + "@" + host, "python3 -c " + shlex.quote(PROBE) + " " + state]
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
            if p.returncode:
                raise RuntimeError(f"rank {rank} snapshot failed: {p.stderr[-400:]}")
            out.append(dict(rank=rank, host=host, **json.loads(p.stdout)))
        (self.d / f"nodes-{label}.json").write_text(json.dumps(out, indent=2))
        bad = [x for x in out if x["sha"] != self.sha or x["swap_kib"] != 0]
        if bad:
            raise RuntimeError(f"swap / binary check failed: {bad}")
        return out

    def down(self):
        if not (self.d / "up.log").exists():
            return
        if self.status.get("ok"):
            try:
                self.snapshot("after")
            except Exception as e:  # noqa: BLE001 — the teardown must still happen
                self.status["ok"] = False
                self.status["error"] = str(e)[-800:]
        run(self.launcher("down", []), self.d / "down.log", self.env, check=False)
        ops = sorted(self.world.glob("serve_rank*.ops"))
        hashes = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in ops}
        match = len(hashes) == self.world_size and len(set(hashes.values())) == 1
        (self.d / "opstreams.json").write_text(json.dumps({"hashes": hashes, "expected_ranks": self.world_size, "match": match}, indent=2))
        self.status["opstreams_match"] = match


def run_campaign(a):
    env = scoped_env(a.no_swap_scope)
    out = pathlib.Path(a.out)
    cfg = deployment_from(a.config, a.port)
    engine = cfg["engine"]
    phases = set(a.phases.split(","))
    unknown = phases - set(PHASES)
    if unknown:
        raise SystemExit(f"unknown phases {sorted(unknown)}; known: {', '.join(PHASES)}")
    slots = int(engine.get("max_concurrency", 8))
    label = a.label or pathlib.Path(a.config).name.replace(".example.json", "").replace(".json", "")
    status = {"config": a.config, "label": label, "started": utcnow(), "phases": sorted(phases)}
    port = str(a.port)
    kw = dict(binary=a.bin)

    boot = Boot(out, cfg, env, **kw)
    host = boot.host
    if a.dry_run:
        print("launcher:", " ".join(shlex.quote(str(c)) for c in boot.launcher("up", [])))
        print(f"clients against {host}:{port}; phases {sorted(phases)}; plain boot {'yes' if drafts(engine) else 'not needed'}; modes {a.mode or []}")
        return 0
    try:
        boot.up()
        if "serving" in phases:
            run(["python3", "scripts/timed_load.py", host, port, "--concurrency", "1,2,4,8", "--classes", "all",
                 "--max-tokens", "256", "--repeat", "3", "--json-out", out / "decode.json"], out / "decode.log", env)
        if "prefill" in phases:
            run(["python3", "scripts/serve_prefill_probe.py", host, port, *map(str, PREFILL_TARGETS), "--repeat", "3",
                 "--no-think", "--tag", f"campaign-{label}-{status['started'][:19]}", "--json-out", out / "prefill.json"],
                out / "prefill.log", env)
        if "context" in phases:
            py = ROOT / ".venv/bin/python"
            run([str(py) if py.exists() else "python3", "scripts/serve_context_matrix.py", host, port, "--model", cfg["model"],
                 "--tag", f"ctx-{label}-{status['started'][:19]}", "--json-out", out / "context.json"], out / "context.log", env,
                timeout=6 * 3600)
        if "quality" in phases:
            run(["python3", "scripts/serve_eval.py", host, port, "--out", out / "eval", "--tasks", "humaneval,gsm8k,extract",
                 "--limit", "300", "--concurrency", str(slots), "--allow-code-execution", "--data", "data"], out / "eval.log", env,
                timeout=4 * 3600)
        if "api" in phases:
            run(["python3", "scripts/serve_api_check.py", host, port], out / "api.log", env, check=False)
        if "transcripts" in phases:
            run(["python3", "scripts/serve_load.py", host, port, "--concurrency", "1", "--classes", "all", "--repeat", "3",
                 "--max-tokens", "256", "--json-out", out / "transcripts_default.json"], out / "transcripts_default.log", env)
            # The isolation check exits nonzero when the texts differ: a result, not a failure.
            run(["python3", "scripts/serve_load.py", host, port, "--concurrency", "1", "--repeat", "1", "--max-tokens", "256",
                 "--isolation", str(slots), "--json-out", out / "isolation.json"], out / "isolation.log", env, check=False)
        boot.status["ok"] = True
    except Exception as e:  # noqa: BLE001 — record, tear down, continue to the other boots
        boot.status["ok"] = False
        boot.status["error"] = str(e)[-800:]
        print("BOOT FAILED", label, boot.status["error"], flush=True)
    finally:
        boot.down()
    status.update(boot.status)

    if "plain" in phases:
        if drafts(engine):
            plain = Boot(out / "plain", plain_deployment(cfg), env, **kw)
            try:
                plain.up()
                run(["python3", "scripts/serve_load.py", host, port, "--concurrency", "1", "--classes", "all", "--repeat", "3",
                     "--max-tokens", "256", "--json-out", out / "plain" / "transcripts_plain.json"], out / "plain" / "transcripts_plain.log",
                    env, timeout=3 * 3600)
                plain.status["ok"] = True
            except Exception as e:  # noqa: BLE001
                plain.status["ok"] = False
                plain.status["error"] = str(e)[-800:]
                print("PLAIN BOOT FAILED", label, plain.status["error"], flush=True)
            finally:
                plain.down()
            status["plain"] = {"deployment": "the template without its drafting keys, mtp false", **plain.status}
        else:
            status["plain"] = {"skipped": "the template drafts nothing: its shipped mode is plain decode"}

    for spec in a.mode or []:
        name, _, knobs = spec.partition("=")
        mode = Boot(out / "modes" / name, cfg, env, knobs=knobs, **kw)
        try:
            mode.up()
            run(["python3", "scripts/timed_load.py", host, port, "--concurrency", "1", "--classes", "all", "--max-tokens", "256",
                 "--repeat", "3", "--json-out", out / "modes" / name / "decode.json"], out / "modes" / name / "decode.log", env)
            mode.status["ok"] = True
        except Exception as e:  # noqa: BLE001
            mode.status["ok"] = False
            mode.status["error"] = str(e)[-800:]
            print("MODE BOOT FAILED", name, mode.status["error"], flush=True)
        finally:
            mode.down()
        status.setdefault("modes", {})[name] = {"knobs": knobs, **mode.status}

    status["finished"] = utcnow()
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print("CAMPAIGN DONE", json.dumps({k: v for k, v in status.items() if k in ("label", "ok", "opstreams_match", "plain", "modes")}))
    return 0 if status.get("ok") else 1


# ---- reading the records ---------------------------------------------------------------------------------

def span(d):
    return f"{min(d.values()):.1f}–{max(d.values()):.1f}"


def serving_cells(d):
    """Per class and concurrency: the engine's decode-only rate and the wall rate (medians of the repetitions)."""
    phases = json.loads((d / "decode.json").read_text())["phases"]
    cs = sorted({p["concurrency"] for p in phases})
    decode = {n: {c: statistics.median(p["engine"]["tokens_per_s"] for p in phases if p["class"] == c and p["concurrency"] == n)
                  for c in CLASSES} for n in cs}
    wall = {n: {c: statistics.median(p["metrics"]["wall_tokens_per_s"] for p in phases if p["class"] == c and p["concurrency"] == n)
                for c in CLASSES} for n in cs}
    return dict(concurrencies=cs, decode=decode, wall=wall)


def prefill_cells(d):
    cold = json.loads((d / "prefill.json").read_text())["samples"]
    targets = sorted({s["requested_tokens"] for s in cold})
    return dict(prefill={n: statistics.median(s["prefill_ms"] for s in cold if s["requested_tokens"] == n) / 1000 for n in targets},
                actual={n: sorted(s["usage"]["prompt_tokens"] for s in cold if s["requested_tokens"] == n) for n in targets},
                cached=sum(s["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for s in cold))


def context_cells(d):
    rep = json.loads((d / "context.json").read_text())
    out = {}
    for n in BUCKETS:
        s = [x for x in rep["samples"] if x["requested_tokens"] == n]
        if not s:
            out[n] = None
            continue
        cold = [x for x in s if x["repeat"] == 0][0]
        out[n] = dict(prefill=cold["engine"]["prefill_ms"] / 1000,
                      ms_per_pass=statistics.median(x["ms_per_pass"] for x in s),
                      tok_s=statistics.median(x["engine_tokens_per_s"] for x in s),
                      cached_cold=cold["answer"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0))
    return dict(buckets=out, limit=next((k["limit"] for k in rep.get("skipped", [])), None))


def eval_cells(d):
    lines = (d / "eval.log").read_text().strip().splitlines()
    return json.loads([l for l in lines if l.startswith("{")][-1])


def api_cells(d):
    lines = (d / "api.log").read_text().splitlines()
    ok = sum(1 for l in lines if l.startswith("ok  "))
    return ok, ok + sum(1 for l in lines if l.startswith("FAIL"))


def class_texts(path):
    rep = json.loads(path.read_text())
    out = {}
    for ph in rep["phases"]:
        if ph["concurrency"] == 1:
            out.setdefault(ph["class"], []).extend((r["text"], r["tokens"]) for r in ph["requests"])
    return out


def transcript_cells(d):
    """Plain/default greedy match per class (every repetition in both modes identical) and the isolation verdict."""
    default = class_texts(d / "transcripts_default.json")
    plain_path = d / "plain" / "transcripts_plain.json"
    plain = class_texts(plain_path) if plain_path.exists() else None
    matched = 0
    detail = {}
    for cls in CLASSES:
        xs = default.get(cls, []) + (plain.get(cls, []) if plain else [])
        same = plain is not None and len(xs) == 6 and len(set(xs)) == 1
        matched += same
        detail[cls] = dict(default_distinct=len(set(default.get(cls, []))),
                           plain_distinct=len(set(plain.get(cls, []))) if plain else None, n=len(xs))
    iso_log = (d / "isolation.log").read_text() if (d / "isolation.log").exists() else ""
    iso_line = next((l for l in iso_log.splitlines() if l.startswith("== isolation")), None)
    return dict(matched=matched if plain is not None else None, detail=detail, plain_measured=plain is not None,
                isolation=None if iso_line is None else ("different" if "transcript isolation failed" in iso_log else "identical"),
                isolation_line=iso_line)


def mode_cells(d):
    out = {}
    for md in sorted((d / "modes").glob("*/decode.json")) if (d / "modes").exists() else []:
        cells = serving_cells(md.parent)
        out[md.parent.name] = cells["decode"][1]
    return out


def validity(d):
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    swaps = {}
    for f in ("nodes-before.json", "nodes-after.json", "plain/nodes-after.json"):
        if (d / f).exists():
            swaps[f] = [x["swap_kib"] for x in json.loads((d / f).read_text())]
    return dict(status=st, swaps=swaps,
                source=(d / "source.txt").read_text().strip()[:12] if (d / "source.txt").exists() else None,
                binary=(d / "binary.sha256").read_text().strip()[:16] if (d / "binary.sha256").exists() else None)


def summarize(d):
    d = pathlib.Path(d)
    v = validity(d)
    st = v["status"]
    print(f"== {d.name}: source {v['source']} binary {v['binary']} ok={st.get('ok')} opstreams={st.get('opstreams_match')} swap={v['swaps']}")
    if (d / "decode.json").exists():
        s = serving_cells(d)
        print("  decode tok/s (engine, decode steps only), range across classes: " + "  ".join(f"C{n} {span(s['decode'][n])}" for n in s["concurrencies"]))
        print("  wall tok/s (prefill and queueing included):                       " + "  ".join(f"C{n} {span(s['wall'][n])}" for n in s["concurrencies"]))
        print("  C1 decode by class: " + ", ".join(f"{c} {s['decode'][1][c]:.1f}" for c in CLASSES))
    if (d / "prefill.json").exists():
        p = prefill_cells(d)
        print("  cold prefill s: " + " / ".join(f"{p['prefill'][n]:.3f}" for n in sorted(p["prefill"])) + f"  (targets {sorted(p['prefill'])}, cached tokens {p['cached']}, actual {p['actual']})")
    if (d / "context.json").exists():
        c = context_cells(d)
        print("  context: " + " | ".join(f"{n}: pf {b['prefill']:.1f} s, {b['ms_per_pass']:.2f} ms/pass, {b['tok_s']:.1f} tok/s, cold cached {b['cached_cold']}" if b else f"{n}: Limit"
                                          for n, b in c["buckets"].items()))
    if (d / "eval.log").exists():
        e = eval_cells(d)
        print("  quality: " + ", ".join(
            f"{k} {x['passed']}/{x['n']} (cap {x['truncated']}" +
            (f", retried {x['retry_completed']} completed / {x['retry_passed']} pass" if x.get('retried') else "") + ")"
            for k, x in e.items()))
    if (d / "api.log").exists():
        print("  api checks: %d/%d" % api_cells(d))
    if (d / "transcripts_default.json").exists():
        t = transcript_cells(d)
        match = f"{t['matched']}/5 classes" if t["plain_measured"] else "not measured (no plain boot)"
        print(f"  plain/default greedy match: {match}; isolation: {t['isolation']}")
    for name, cells in mode_cells(d).items():
        print(f"  mode {name}: C1 decode {span(cells)} ({', '.join(f'{c} {cells[c]:.1f}' for c in CLASSES)})")


def rows(d, label, nodes, options):
    """The doc's rows for this deployment, in table order; the author fills the model label and options once."""
    d = pathlib.Path(d)
    head = f"| {label} | {nodes} | {options} |"
    cfg = json.loads((d / "deployment.json").read_text()) if (d / "deployment.json").exists() else {}
    e = cfg.get("engine", {})
    out = []
    if (d / "decode.json").exists():
        s = serving_cells(d)
        pf = prefill_cells(d)["prefill"] if (d / "prefill.json").exists() else {}
        out.append(("serving", f"{head} " + " | ".join(span(s["decode"][n]) for n in (1, 2, 4, 8))
                    + " | " + (" / ".join(f"{pf[n]:.3f}" for n in (2048, 8192, 32768)) if pf else "—") + " |"))
        out.append(("wall", f"{head} " + " | ".join(span(s["wall"][n]) for n in (1, 2, 4, 8)) + " |"))
        out.append(("classes", f"{head} " + " | ".join(f"{s['decode'][1][c]:.1f}" for c in CLASSES) + " |"))
    if e:
        if e.get("dflash_model"):
            mode = "DFlash2 adaptive ≤7" if e.get("mtp_schedule") else "DFlash2 fixed 7"
        elif e.get("mtp"):
            mode = f"MTP{e.get('mtp_depth', 1)}" + (" scheduled" if e.get("mtp_schedule") else "")
        else:
            mode = "plain"
        budget = "adaptive (default)" if "prefill_budget_tokens" not in e else f"{e['prefill_budget_tokens']} / {e.get('prefill_idle_budget_tokens', e['prefill_budget_tokens'])}"
        out.append(("config", f"{head} {e.get('max_concurrency')} | {int(e.get('kv_capacity', 0)):,} | {e.get('kv_dtype', 'bf16')} | "
                    f"{float(e.get('prefix_cache_gib', 1.5)):g} | {budget} | {mode} | [JSON](<template path>) |"))
    if (d / "context.json").exists():
        c = context_cells(d)
        b = c["buckets"]
        cells = []
        for n in BUCKETS:
            cells += [f"{b[n]['ms_per_pass']:.2f}", f"{b[n]['tok_s']:.1f}"] if b[n] else ["Limit", "Limit"]
        limit = f"{c['limit']:,}" if c["limit"] else "—"
        out.append(("context", f"{head} {limit} | " + (f"{b[32768]['prefill']:.3f}" if b.get(32768) else "—") + " | " + " | ".join(cells) + " |"))
        out.append(("context-prefill", f"{head} " + " | ".join(f"{b[n]['prefill']:.3f}" if b[n] else "Limit" for n in BUCKETS) + " |"))
    if (d / "eval.log").exists():
        ev = eval_cells(d)
        # The last column: of the responses at the standard cap, how many
        # completed when asked once more at the larger cap, and how many of
        # those passed (serve_eval.py --retry-max-tokens; "—" before 2026-10-22).
        retried = sum(x.get("retried", 0) for x in ev.values())
        beyond = (f"{sum(x['retry_completed'] for x in ev.values())} ({sum(x['retry_passed'] for x in ev.values())} pass)"
                  if retried else ("0" if sum(x['truncated'] for x in ev.values()) == 0 else "—"))
        out.append(("quality", f"{head} {ev['humaneval']['passed']}/{ev['humaneval']['n']} | {ev['gsm8k']['passed']}/{ev['gsm8k']['n']} | "
                    f"{ev['extract']['passed']}/{ev['extract']['n']} | {sum(x['truncated'] for x in ev.values())} | {beyond} |"))
    if (d / "transcripts_default.json").exists() and (d / "api.log").exists():
        t = transcript_cells(d)
        ok, total = api_cells(d)
        match = f"{t['matched']}/5 classes" if t["plain_measured"] else "—"
        out.append(("transcripts", f"{head} {ok}/{total} | {match} | {t['isolation'] or '—'} |"))
    m = mode_cells(d)
    if m:
        out.append(("modes", f"{head} " + " | ".join(f"{name}: {span(cells)}" for name, cells in m.items()) + " |"))
    for table, row in out:
        print(f"-- {table}\n{row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run", help="one template's whole row set")
    r.add_argument("--config", required=True, help="the deployment template (deploy/*.json)")
    r.add_argument("--out", required=True, help="the record directory (new; benchmarks/results/<label> is ignored by Git)")
    r.add_argument("--bin", default=str(ROOT / "build-release/dgpp-serve"), help="the binary under test")
    r.add_argument("--port", type=int, default=18081)
    r.add_argument("--label", default=None)
    r.add_argument("--phases", default=",".join(PHASES))
    r.add_argument("--mode", action="append", metavar='NAME="<knobs>"', help="an extra decode-mode boot: the C1 sweep under these knobs")
    r.add_argument("--no-swap-scope", action="store_true", help="do not run the ranks in the launcher's swap-limited scopes")
    r.add_argument("--dry-run", action="store_true", help="print the launcher command for this leg and exit")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    w = sub.add_parser("rows")
    w.add_argument("dir")
    w.add_argument("--label", required=True)
    w.add_argument("--nodes", required=True)
    w.add_argument("--options", required=True)
    a = ap.parse_args()
    if a.verb == "run":
        sys.exit(run_campaign(a))
    elif a.verb == "summarize":
        summarize(a.dir)
    else:
        rows(a.dir, a.label, a.nodes, a.options)


if __name__ == "__main__":
    main()
