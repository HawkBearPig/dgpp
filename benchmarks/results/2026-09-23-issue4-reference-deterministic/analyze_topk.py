#!/usr/bin/env python3
"""Audit real reference scores and repeated native selections on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def analyze(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    scores = data["logits"].numpy()
    lengths = data["lengths"].numpy()
    outputs = [t.numpy() for t in data["outputs"]]
    k = outputs[0].shape[1]
    result = {"path": str(path), "rows": len(lengths), "order_variable_rows": 0,
              "set_variable_rows": 0, "invalid_rows": 0, "wrong_value_rows": 0,
              "tied_threshold_rows": 0, "canonical_set_mismatch_rows": 0}
    for row, n in enumerate(lengths):
        values = scores[row, :n]
        assert np.isfinite(values).all(), (path, row)
        take = min(n, k)
        selected = [out[row, :take] for out in outputs]
        result["order_variable_rows"] += int(any(
            not np.array_equal(selected[0], s) for s in selected[1:]))
        result["set_variable_rows"] += int(any(
            not np.array_equal(np.sort(selected[0]), np.sort(s)) for s in selected[1:]))
        valid = all((s >= 0).all() and (s < n).all()
                    and len(np.unique(s)) == take for s in selected)
        result["invalid_rows"] += int(not valid)
        if not take or not valid:
            continue
        canonical = np.lexsort((np.arange(n), -values))[:take]
        threshold = values[canonical[-1]]
        result["tied_threshold_rows"] += int(n > k and (values == threshold).sum() > 1)
        result["wrong_value_rows"] += int(any(
            not np.array_equal(np.sort(values[s]), np.sort(values[canonical])) for s in selected))
        result["canonical_set_mismatch_rows"] += int(any(
            not np.array_equal(np.sort(s), np.sort(canonical)) for s in selected))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = [analyze(p) for p in sorted(args.captures.glob("rank*/request*/position*/qsa_topk_*.pt"))]
    assert results
    totals = {key: sum(r[key] for r in results)
              for key in results[0] if key != "path"}
    parity = []
    for left in sorted(args.captures.glob("rank*/request1/position*/qsa_topk_*.pt")):
        right = Path(str(left).replace("/request1/", "/request2/"))
        a = torch.load(left, map_location="cpu", weights_only=True)
        b = torch.load(right, map_location="cpu", weights_only=True)
        parity.append({"path": str(left), "same_scores": torch.equal(a["logits"], b["logits"]),
                       "same_lengths": torch.equal(a["lengths"], b["lengths"])})
    args.output.write_text(json.dumps({"totals": totals, "files": results, "input_parity": parity}, indent=2) + "\n")
    print(json.dumps({"totals": totals, "all_repeated_inputs_identical": all(
        r["same_scores"] and r["same_lengths"] for r in parity)}, indent=2))
