#!/usr/bin/env python3
"""Summarize exact-input captures; no inference or service changes."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CAPTURES = RAW / "captures"


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def bf16(path):
    return (np.fromfile(path, dtype="<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)


audits = json.loads((RAW / "operator-audit.json").read_text())
assert len(audits) == 96
summary = {}
for kind in ("gdn", "qsa"):
    rows = [r for r in audits if r["kind"] == kind]
    stats = {"captures": len(rows)}
    for key, value in rows[0].items():
        if isinstance(value, dict):
            worst = max(rows, key=lambda r: r[key]["relative_l2"])
            stats[key] = {
                "max_relative_l2": worst[key]["relative_l2"],
                "worst_layer": "/".join(Path(worst["path"]).parts[-2:]),
                "max_relative_two_ulp_mismatch_fraction": max(r[key]["relative_two_ulp_mismatches"] / r[key]["n"] for r in rows),
                "nonfinite_count": sum(r[key]["nonfinite"] for r in rows),
            }
        elif isinstance(value, bool):
            stats[key] = all(r[key] for r in rows)
        elif isinstance(value, int):
            stats[key] = sum(r[key] for r in rows)
    summary[kind] = stats

comparisons = []
for layer in range(48):
    names = ["attention_input.bin", "attention_folded.bin", "post_mlp_residual.bin"]
    if layer % 4 == 3:
        names += ["scores.bin", "selected.bin", "index_cache.bin", "qi.bin"]
    for name in names:
        paths = [CAPTURES / f"rank{r}/layer{layer}" / name for r in range(2)]
        comparisons.append({"layer": layer, "field": name, "bitwise_equal": digest(paths[0]) == digest(paths[1])})
summary["cross_rank"] = {"comparisons": len(comparisons), "all_bitwise_equal": all(c["bitwise_equal"] for c in comparisons), "mismatches": [c for c in comparisons if not c["bitwise_equal"]]}

clean = json.loads((RAW / "clean-response.json").read_text())
capture = json.loads((RAW / "capture-response.json").read_text())
summary["prediction"] = {"choices_equal": clean["choices"] == capture["choices"], "usage_equal": clean["usage"] == capture["usage"], "choices": clean["choices"], "usage": clean["usage"]}
tokens = json.loads((RAW / "forced-prefix.ids.json").read_text())
summary["token_sequence"] = {"length": len(tokens), "int64_le_sha256": hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()}
assert summary["token_sequence"] == {"length": 261290, "int64_le_sha256": "fe207c412e6a70d70b00bbc3c5995488ed12ce964c4ac7eb90933ab61e82785d"}

ranges = json.loads((ROOT / "record-token-ranges.json").read_text())
retrieval = []
for rank in range(2):
    for layer in range(3, 48, 4):
        p = CAPTURES / f"rank{rank}/layer{layer}"
        fields = (p / "qsa.meta").read_text().split()
        T, H, KV, D = map(int, fields[:4])
        assert KV == 1
        scale = float(np.fromfile(p / "qsa_scalars.bin", dtype="<f4")[1])
        selected = np.fromfile(p / "selected.bin", dtype="<i4")
        q, k = bf16(p / "qn.bin").reshape(H, D), bf16(p / "k_cache.bin").reshape(len(selected), D)
        scores = (q @ k.T) * scale
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        records = {}
        for key, interval in ranges.items():
            mask = (selected >= interval["first_token"]) & (selected <= interval["last_token"])
            mass = probabilities[:, mask].sum(axis=1)
            records[key] = {"selected_tokens": selected[mask].tolist(), "attention_mass_by_head_fp64": mass.tolist(), "mean_mass": float(mass.mean()), "max_head_mass": float(mass.max())}
        retrieval.append({"rank": rank, "layer": layer, "records": records})

(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(ROOT / "retrieval.json").write_text(json.dumps(retrieval, indent=2) + "\n")
print(json.dumps(summary, indent=2))
for r in retrieval:
    print("rank", r["rank"], "layer", r["layer"], "correct", len(r["records"]["key_0769e0226c63"]["selected_tokens"]), r["records"]["key_0769e0226c63"]["max_head_mass"], "wrong", len(r["records"]["key_def01c9d0367"]["selected_tokens"]), r["records"]["key_def01c9d0367"]["max_head_mass"])
