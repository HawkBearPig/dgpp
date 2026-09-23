#!/usr/bin/env python3
"""Independent CPU replay of the captured final row's attention GR mixer.

Consumes the preceding layer's captured residual, except layer zero, which
starts from its checkpoint embedding. The PLE layer is excluded because its
update was not captured. This is a boundary replay, not a history reference.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--captures", type=Path, required=True)
parser.add_argument("--tokens", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
# Checkpoint PLE layer IDs are one-based; capture directory IDs are zero-based.
ple_layers = {layer - 1 for layer in config["ple_layer_ids"]}
assert ple_layers == {layer for layer in range(48) if any(name.startswith(f"model.language_model.layers.{layer}.ple.") for name in index)}
headers, hashes = {}, {}
tokens = json.loads(args.tokens.read_text())
assert len(tokens) == 261290
assert hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest() == "fe207c412e6a70d70b00bbc3c5995488ed12ce964c4ac7eb90933ab61e82785d"


def bf(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def rb(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return bf((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16)


def weight(name, row=None):
    file = args.checkpoint / index[name]
    if file not in headers:
        with file.open("rb") as f:
            size, = struct.unpack("<Q", f.read(8))
            headers[file] = size + 8, json.loads(f.read(size))
    offset, metadata = headers[file]
    entry = metadata[name]
    assert entry["dtype"] == "BF16"
    bits = np.memmap(file, dtype="<u2", mode="r", offset=offset + entry["data_offsets"][0], shape=tuple(entry["shape"]))
    if row is not None:
        bits = bits[row]
    hashes[name + (f"[row={row}]" if row is not None else "")] = hashlib.sha256(bits.tobytes()).hexdigest()
    return bf(bits)


def fp8(w):
    nr, nc = w.shape
    # Zero padding leaves each incomplete 128x128 block's amax unchanged.
    padded = np.pad(w, ((0, (-nr) % 128), (0, (-nc) % 128)))
    pr, pc = padded.shape
    blocks = padded.reshape(pr // 128, 128, pc // 128, 128)
    maximum = np.abs(blocks).max(axis=(1, 3), keepdims=True)
    scale = np.where(maximum > 0, maximum / np.float32(448), np.float32(1))
    scaled = blocks / scale
    _, exponent = np.frexp(np.abs(scaled))
    step = np.exp2(np.maximum(exponent - 4, -9)).astype(np.float32)
    codes = np.clip(np.rint(scaled / step) * step, -448, 448)
    return rb(codes * scale).reshape(pr, pc)[:nr, :nc]


def last(rank, layer, field, width):
    p = args.captures / f"rank{rank}/layer{layer}/{field}.bin"
    with p.open("rb") as f:
        f.seek(-width * 2, 2)
        return bf(np.frombuffer(f.read(width * 2), dtype="<u2"))


def sigmoid(x):
    # FP64 transcendental, followed by the model's explicit BF16 staging.
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))


results = []
for layer in range(48):
    if layer in ple_layers:
        continue
    if layer == 0:
        residual = np.tile(weight("model.language_model.embed_tokens.weight", tokens[-1]), 4)
    else:
        residual = last(0, layer - 1, "post_mlp_residual", 10240)
        assert np.array_equal(residual, last(1, layer - 1, "post_mlp_residual", 10240))
    p = f"model.language_model.layers.{layer}.attn_hyper_connection."
    r = residual.reshape(4, 2560)
    wnorm = weight(p + "hc_norm.weight").reshape(4, 2560)
    rs = (1 / np.sqrt(np.mean(r.astype(np.float64)**2, axis=1, keepdims=True) + np.float32(1e-6))).astype(np.float32)
    normalized = rb((r * rs) * (np.float32(1) + wnorm)).reshape(-1)
    down = fp8(weight(p + "input_mix_weight_down.weight"))
    reduced = rb(down.astype(np.float64) @ normalized.astype(np.float64))
    activated = rb((reduced / 4) * sigmoid(reduced / 4))
    up = fp8(weight(p + "input_mix_weight_up.weight"))
    logits = rb(up.astype(np.float64) @ activated.astype(np.float64)).reshape(4, 2560)
    gated = rb(rb(sigmoid(logits)) * normalized.reshape(4, 2560))
    result = rb(gated.sum(axis=0, dtype=np.float32) / 4)
    for rank in range(2):
        actual = last(rank, layer, "attention_input", 2560)
        diff = np.abs(result.astype(np.float64) - actual)
        relative_bad = diff > np.maximum(1e-7, np.maximum(np.abs(result), np.abs(actual)) * (2 / 128))
        entry = {"layer": layer, "rank": rank, "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(result.astype(np.float64)), np.linalg.norm(actual.astype(np.float64)), 1e-30)), "max_abs": float(diff.max()), "two_bf16_relative_mismatches": int(relative_bad.sum()), "rms": float(np.sqrt(np.mean(result.astype(np.float64)**2))), "nonfinite": int((~np.isfinite(result) | ~np.isfinite(actual)).sum())}
        results.append(entry)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("layer", layer, "relative L2", results[-1]["relative_l2"], "relative mismatches", results[-1]["two_bf16_relative_mismatches"], flush=True)

summary = {"comparisons": len(results), "worst_relative_l2": max(results, key=lambda r: r["relative_l2"]), "two_bf16_relative_mismatches": sum(r["two_bf16_relative_mismatches"] for r in results), "nonfinite": sum(r["nonfinite"] for r in results), "excluded_layers": {str(layer): "PLE intervenes before the attention mixer; post-PLE residual not captured" for layer in sorted(ple_layers)}, "scope": "Final-row attention mixer only; input residuals are captured production values."}
(args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(args.out / "tensor-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
