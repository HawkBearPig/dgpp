#!/usr/bin/env python3
"""Replay final-row dense attention projections from checkpoint BF16 tensors.

This implements E4M3 quantization arithmetically in NumPy, independently of
DGPP's loader/decoder, and accumulates the reference products in FP64.
"""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CHECKPOINT = Path("/home/stephen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47")
INDEX = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
HEADERS = {}
TENSOR_HASHES = {}
HIDDEN = 2560
ROWS = 1194


def bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def round_bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return bf16((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16)


def tensor(name):
    file = CHECKPOINT / INDEX[name]
    if file not in HEADERS:
        with file.open("rb") as f:
            size, = struct.unpack("<Q", f.read(8))
            HEADERS[file] = size + 8, json.loads(f.read(size))
    offset, header = HEADERS[file]
    item = header[name]
    assert item["dtype"] == "BF16"
    bits = np.memmap(file, mode="r", dtype="<u2", offset=offset + item["data_offsets"][0], shape=tuple(item["shape"]))
    if name not in TENSOR_HASHES:
        TENSOR_HASHES[name] = hashlib.sha256(bits.tobytes()).hexdigest()
    return bf16(bits)


def block_fp8(weight):
    weight = np.ascontiguousarray(weight)
    nr, nc = weight.shape
    assert nr % 128 == nc % 128 == 0
    blocks = weight.reshape(nr // 128, 128, nc // 128, 128)
    maximum = np.abs(blocks).max(axis=(1, 3), keepdims=True)
    scale = np.where(maximum > 0, maximum / np.float32(448), np.float32(1))
    scaled = blocks / scale
    _, exponent = np.frexp(np.abs(scaled))
    step = np.exp2(np.maximum(exponent - 4, -9)).astype(np.float32)
    codes = np.clip(np.rint(scaled / step) * step, -448, 448)
    return round_bf16(codes * scale).reshape(nr, nc)


def captured(path, name, width, row=-1):
    return bf16(np.fromfile(path / (name + ".bin"), dtype="<u2").reshape(-1, width)[row])


def product(weight, x, quantized=True):
    w = block_fp8(weight) if quantized else weight
    return round_bf16(w.astype(np.float64) @ x.astype(np.float64))


def statistics(ref, actual):
    ref, actual = ref.astype(np.float64), actual.astype(np.float64)
    error = np.abs(ref - actual)
    rms = np.sqrt(np.mean(ref * ref))
    mismatch = error > np.maximum(np.float64(1e-7), np.maximum(np.abs(ref), np.abs(actual)) * (2 / 128))
    return {"n": len(ref), "relative_l2": float(np.linalg.norm(ref - actual) / max(np.linalg.norm(ref), np.linalg.norm(actual), 1e-30)), "max_abs": float(error.max()), "two_bf16_relative_mismatches": int(mismatch.sum()), "over_two_percent_rms_and_two_bf16_relative": int((mismatch & (error > .02 * rms)).sum()), "nonfinite": int((~np.isfinite(ref) | ~np.isfinite(actual)).sum())}


results = []
for layer in range(48):
    output_partials = []
    for rank in range(2):
        path = RAW / "captures" / f"rank{rank}/layer{layer}"
        x = captured(path, "attention_input", HIDDEN)
        prefix = f"model.language_model.layers.{layer}."
        checks = {}
        if layer % 4 != 3:
            prefix += "linear_attn."
            w = tensor(prefix + "in_proj_qkv.weight")
            local = np.concatenate([w[rank*1024:(rank+1)*1024], w[2048+rank*1024:2048+(rank+1)*1024], w[4096+rank*3072:4096+(rank+1)*3072]])
            checks["qkv"] = statistics(product(local, x), captured(path, "qkv", 5120))
            for name, width, quantized in (("z", 3072, True), ("a", 24, False), ("b", 24, False)):
                w = tensor(prefix + f"in_proj_{name}.weight")[rank*width:(rank+1)*width]
                checks[name] = statistics(product(w, x, quantized), captured(path, name, width))
            w = tensor(prefix + "out_proj.weight")[:, rank*3072:(rank+1)*3072]
            output_partials.append(product(w, captured(path, "normed", 3072)))
        else:
            prefix += "self_attn."
            for name, parameter, width, sharded in (("q", "q_proj", 6144, True), ("idx", "indexer.index_qk_proj", 640, False), ("v_tail", "v_proj", 256, True)):
                w = tensor(prefix + parameter + ".weight")
                if sharded:
                    w = w[rank*width:(rank+1)*width]
                checks[name] = statistics(product(w, x), captured(path, name, width))
            w = tensor(prefix + "o_proj.weight")[:, rank*3072:(rank+1)*3072]
            output_partials.append(product(w, captured(path, "attention_gated", 3072)))
        results.append({"layer": layer, "rank": rank, "projections": checks})
    folded = round_bf16(output_partials[0] + output_partials[1])
    result = {"layer": layer, "rank": "folded", "projections": {"output_projection": statistics(folded, captured(path, "attention_folded", HIDDEN))}}
    results.append(result)
    (RAW / "projection-audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print("layer", layer, "max relative L2", max(v["relative_l2"] for r in results[-3:] for v in r["projections"].values()), flush=True)

(ROOT / "checkpoint-tensor-hashes.json").write_text(json.dumps(TENSOR_HASHES, indent=2) + "\n")
print("All dense projection captures replayed", flush=True)
