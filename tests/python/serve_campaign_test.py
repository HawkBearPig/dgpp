#!/usr/bin/env python3
"""serve_campaign.py's readers on a synthetic record directory: the decode-only
and wall rates per class and concurrency, the cold-prefill medians, the
context cells, the quality summary, the API tally, the plain/default match
and the isolation verdict, and the plain-knobs derivation."""
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import serve_campaign as sc  # noqa: E402


def phase(cls, c, engine_rate, wall_rate, text="x"):
    return {"class": cls, "concurrency": c, "engine": {"tokens_per_s": engine_rate},
            "metrics": {"wall_tokens_per_s": wall_rate}, "requests": [{"text": text, "tokens": 256}] * c}


def write_records(d):
    phases = []
    for cls_i, cls in enumerate(sc.CLASSES):
        for c in (1, 2, 4, 8):
            # three repetitions whose median is 10 * c + cls_i (engine) and 9 * c + cls_i (wall)
            for rep_rate in (-1, 0, 1):
                phases.append(phase(cls, c, 10 * c + cls_i + rep_rate, 9 * c + cls_i + rep_rate))
    (d / "decode.json").write_text(json.dumps({"phases": phases}))
    samples = []
    for n in sc.PREFILL_TARGETS:
        for ms in (n - 1, n, n + 1):
            samples.append({"requested_tokens": n, "prefill_ms": ms, "usage": {"prompt_tokens": n + 5, "prompt_tokens_details": {"cached_tokens": 0}}})
    (d / "prefill.json").write_text(json.dumps({"samples": samples}))
    ctx = []
    for n in (0, 32768):
        for r in range(3):
            ctx.append({"requested_tokens": n, "repeat": r, "engine": {"prefill_ms": 1000 * (n // 32768 + 1)},
                        "ms_per_pass": 100 + r, "engine_tokens_per_s": 20 - r,
                        "answer": {"usage": {"prompt_tokens": n or 100, "prompt_tokens_details": {"cached_tokens": 0 if r == 0 else n}}}})
    (d / "context.json").write_text(json.dumps({"samples": ctx, "skipped": [{"target": 65536, "reason": "exceeds request limit", "limit": 40000}]}))
    (d / "eval.log").write_text("== humaneval: 160/164\n" + json.dumps({"humaneval": {"n": 164, "passed": 160, "truncated": 2},
                                                                         "gsm8k": {"n": 300, "passed": 294, "truncated": 1},
                                                                         "extract": {"n": 100, "passed": 100, "truncated": 0}}) + "\n")
    (d / "api.log").write_text("ok   stop: fine\nok   n: fine\nFAIL usage: missing\n")
    default = {"phases": [phase(cls, 1, 1, 1, text=f"{cls}-answer") for cls in sc.CLASSES for _ in range(3)]}
    (d / "transcripts_default.json").write_text(json.dumps(default))
    (d / "plain").mkdir()
    plain_phases = [phase(cls, 1, 1, 1, text=f"{cls}-answer" if cls != "chat" else "chat-other") for cls in sc.CLASSES for _ in range(3)]
    (d / "plain" / "transcripts_plain.json").write_text(json.dumps({"phases": plain_phases}))
    (d / "isolation.log").write_text("== isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): DIFFERENT\ntranscript isolation failed\n")
    (d / "status.json").write_text(json.dumps({"ok": True, "opstreams_match": True}))
    (d / "nodes-before.json").write_text(json.dumps([{"rank": 0, "swap_kib": 0}]))
    (d / "source.txt").write_text("0123456789abcdef\n")
    (d / "binary.sha256").write_text("f" * 64 + "\n")
    (d / "deployment.json").write_text(json.dumps({"engine": {"max_concurrency": 8, "kv_capacity": 786432, "kv_dtype": "fp8",
                                                              "prefix_cache_gib": 20, "dflash_model": "x", "mtp_schedule": True}}))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        write_records(d)
        s = sc.serving_cells(d)
        assert s["concurrencies"] == [1, 2, 4, 8]
        assert s["decode"][1]["prose"] == 10 and s["decode"][8]["chat"] == 84, s["decode"]
        assert s["wall"][4]["json"] == 38, s["wall"]
        assert sc.span(s["decode"][2]) == "20.0–24.0"
        p = sc.prefill_cells(d)
        assert p["prefill"][2048] == 2.048 and p["cached"] == 0, p
        c = sc.context_cells(d)
        assert c["buckets"][0]["ms_per_pass"] == 101 and c["buckets"][0]["tok_s"] == 19, c["buckets"][0]
        assert c["buckets"][32768]["prefill"] == 2.0 and c["buckets"][65536] is None and c["limit"] == 40000
        e = sc.eval_cells(d)
        assert e["gsm8k"]["passed"] == 294
        assert sc.api_cells(d) == (2, 3)
        t = sc.transcript_cells(d)
        assert t["matched"] == 4 and t["detail"]["chat"]["plain_distinct"] == 1 and t["isolation"] == "different", t
        assert sc.drafts({"dflash_model": "z"}) and sc.drafts({"mtp": True}) and not sc.drafts({"mtp": False})
        plain = sc.plain_deployment({"engine": {"mtp": False, "dflash_model": "z", "mtp_schedule": True, "kv_capacity": 4}})["engine"]
        assert plain == {"mtp": False, "kv_capacity": 4}, plain
        # without a plain boot the match is not a result
        import shutil
        shutil.move(str(d / "plain"), str(d / "plain.off"))
        t2 = sc.transcript_cells(d)
        assert t2["matched"] is None and t2["plain_measured"] is False
        shutil.move(str(d / "plain.off"), str(d / "plain"))
        v = sc.validity(d)
        assert v["source"] == "0123456789ab" and v["swaps"]["nodes-before.json"] == [0]
        # The row writer runs end to end and names every table it has records for.
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sc.rows(d, "Model X", 1, "opts")
        out = buf.getvalue()
        for table in ("serving", "wall", "classes", "config", "context", "context-prefill", "quality", "transcripts"):
            assert f"-- {table}\n| Model X | 1 | opts |" in out, table
        assert "| 786,432 | fp8 | 20 | adaptive (default) | DFlash2 adaptive ≤7 |" in out, out
    print("serve_campaign_test: ok")


if __name__ == "__main__":
    main()
