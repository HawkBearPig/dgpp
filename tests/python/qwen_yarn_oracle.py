#!/usr/bin/env python3
"""The Qwen YaRN goldens' provenance (2026-09-18), three ways.

Print mode (the default): every value tests/unit/qwen_rope_scaling_test.cpp
and tests/cuda/qsa_test.cu freeze — the YaRN inverse-frequency table, the
mrope-enlarged vs native correction band, the attention mscale and the bf16
cos/sin the vLLM cache holds — computed by the SAME vLLM code the user's
recipe runs, not by a re-derivation.

Fixture mode (--emit / --check): the same values, machine-readable, and
more of them — the table and the mscale, the bf16 cos/sin cache rows, and
rotated Q/K rows (plus one attention logit per position) at several
positions, for the recipe AND for non-default parameter combinations,
alongside the digest of the container that produced them:

    .../qwen_yarn_oracle.py /ckpt --emit /d/qwen_yarn_oracle.json
    .../qwen_yarn_oracle.py /ckpt --check /d/qwen_yarn_oracle.json

--emit writes the fixture, --check recomputes every value it records and
exits non-zero on any mismatch. Both need vLLM, so both run in the
container below; NEITHER is needed for a normal build: the C++ gate
tests/unit/qwen_yarn_fixture_test.cpp reads the committed fixture and
compares the ENGINE's table, cos/sin, rotation and mscale against it, with
no vLLM and no GPU in sight. When vLLM's yarn code or the recipe changes,
refresh the fixture here and the C++ gate names the case that drifted.

Run inside the recipe's image (a GPU is present but unused for these host
tensors). The image's digest is recorded in the fixture, so look it up on
the host and hand it in:

  HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4
  #   (this box: /data/hf/hub/... — the recipe's HF_HOME)
  IMG=vllm/vllm-openai:qwen38-flash-next
  DIG=$(docker inspect --format '{{json .RepoDigests}}' "$IMG" | tr -d '[]"' | sed 's/.*@//')
  docker run --rm --gpus all --entrypoint python3 \
    -e DGPP_ORACLE_IMAGE_DIGEST="$DIG" \
    -v "$HUB":/ckpt:ro -v "$PWD/tests/python":/w -v "$PWD/tests/data":/d \
    "$IMG" /w/qwen_yarn_oracle.py \
    /ckpt/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
    --emit /d/qwen_yarn_oracle.json

Print mode needs no mount of tests/data and no digest, as before:

  docker run --rm --gpus all --entrypoint python3 \
    -v "$HUB":/ckpt:ro -v "$PWD/tests/python":/w \
    vllm/vllm-openai:qwen38-flash-next /w/qwen_yarn_oracle.py \
    /ckpt/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47

(The whole models-- directory is mounted, not the snapshot alone:
config.json is a symlink into ../../blobs. The image is the digest-pinned
one in the recipe's .env.)

What it reproduces, in the recipe's own terms:

  * start.sh emits --hf-overrides '{"text_config":{"rope_parameters":
    {"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":
    262144}}}' with VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and MAX_MODEL_LEN=524288,
    and YARN_ENABLE=true / YARN_FACTOR=2.0 in .env.
  * vllm ModelConfig._update_nested MERGES that into the checkpoint's
    existing text_config.rope_parameters, so mrope_section [11,11,10]
    survives and get_rope builds an MRotaryEmbedding (not
    YaRNScalingRotaryEmbedding).
  * MRotaryEmbedding builds its cache at max_position_embeddings * 4
    (mrope.py) and YaRN's correction band is computed from that value:
    low/high = 16/24, where a non-mrope YaRN over 262144 would use 14/22.
    That enlargement is the fixture's `mrope_cache_factor` (4.0 = the mrope
    path, 1.0 = the plain YaRN one). get_rope's yarn branch hands
    original_max_position_embeddings — NOT its own max_position argument —
    to the rope object, so the two factors are the whole story for the band.
  * mscale = yarn_get_mscale(factor) * attn_factor = 1.0693147180559945
    rides the bf16 cos/sin cache; the attention scale stays
    head_dim**-0.5 (nvidia/qsa.py).

The fixture's arithmetic contract, recorded in its "spec" block and
re-implemented lane for lane by the C++ gate. It is what makes a rotated
row a check on vLLM's table rather than on this script:

  * the input rows and their norm weights come from splitmix64 over the
    recorded seed, one draw per lane, straight to bf16 BITS — no float
    rounding involved, so either side regenerates them identically (the
    fixture also records an FNV-1a of the drawn bits, so a spec drift is
    named instead of silently tolerated);
  * the (1+w) RMSNorm in fp32, in lane order: ss += v*v per lane,
    rstd = 1/sqrt(ss/dim + eps), xn = bf16(v*rstd*(1+w));
  * the rotation is the neox half-split (lane d pairs with d + rotary/2)
    against vLLM's bf16 cache row: out = bf16(bf16(xn*c) + bf16(rot*s)),
    and the lanes at or past rotary_dim pass through. The pairing style is
    the engine's; what the fixture makes a golden is the cache row's
    VALUES, which do not depend on it.
  * the logit is the fp32 sum of q[d]*k[d] in lane order times
    head_dim**-0.5 — one number that carries the mscale^2 the rotated
    lanes pick up.
  * nothing is contracted into a fused multiply-add. The repo builds with
    -ffp-contract=off, so the engine's sequence is literally this one.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import struct
import sys

import numpy as np
import torch

FACTOR = 2.0
ORIGINAL = 262144
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
FORMAT = 1

# The deterministic input (see the contract above): two rows of the
# checkpoint's head_dim — row 0 is the fixture's Q, row 1 its K — and the
# positions: the recipe's own, plus each case's last servable one
# (original x factor - 1).
SEED = (0x9E3779B97F4A7C15 ^ 0x20260918) & ((1 << 64) - 1)
ROWS = 2
POSITIONS = (0, 12345, 262143, 524287)

# factor, (beta_fast, beta_slow), attn_factor, mrope_cache_factor. The
# recipe first, then the combinations a non-default knob can take: another
# factor, another correction band, an attention factor that is not 1, and
# the non-mrope (1x) cache.
CASES = (
    ("recipe", 2.0, (32.0, 1.0), 1.0, 4.0),
    ("factor_4", 4.0, (32.0, 1.0), 1.0, 4.0),
    ("betas_16_2", 2.0, (16.0, 2.0), 1.0, 4.0),
    ("attn_factor_1_2", 2.0, (32.0, 1.0), 1.2, 4.0),
    ("mrope_cache_1", 2.0, (32.0, 1.0), 1.0, 1.0),
)

U64 = (1 << 64) - 1


class SplitMix64:
    """The one PRNG both sides implement (splitmix64, the version in the
    C++ fixture gate — 64-bit state, add the golden ratio, two multiplies
    and three shifts)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & U64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & U64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & U64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & U64
        return (z ^ (z >> 31)) & U64


def fixture_input(seed: int, rows: int, dim: int):
    """x[rows][dim] and w[rows][dim] as bf16 bit patterns, one draw a lane.

    The x lanes are random finite bf16 normals: a random sign, exponent
    124..128 (so |x| in [1/8, 4)) and a random 7-bit mantissa. The w lanes
    are small (exponent 125), so 1+w stays in (1/2, 3/2) and the normalized
    row stays in a range where bf16 has room.
    """
    rng = SplitMix64(seed)
    x, w = [], []
    for _ in range(rows * dim):
        v = rng.next()
        x.append((((v >> 63) & 1) << 15) | ((124 + (v % 5)) << 7) | ((v >> 11) & 0x7F))
    for _ in range(rows * dim):
        v = rng.next()
        w.append((((v >> 63) & 1) << 15) | (125 << 7) | ((v >> 11) & 0x7F))
    return [x[r * dim : (r + 1) * dim] for r in range(rows)], [
        w[r * dim : (r + 1) * dim] for r in range(rows)
    ]


def bits_to_f32(u: int) -> np.float32:
    """An fp32 bit pattern as an fp32 scalar, with no re-rounding."""
    return np.frombuffer(struct.pack("<I", u & 0xFFFFFFFF), dtype=np.float32)[0]


def f32_bits(v) -> int:
    return int(np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0])


def bf16_bits_to_f32(u: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", (u & 0xFFFF) << 16), dtype=np.float32)[0]


def f32_to_bf16_bits(v) -> int:
    """Round-to-nearest-even to bf16 — the integer trick dtypes.hpp uses."""
    u = f32_bits(v)
    if (u & 0x7FFFFFFF) > 0x7F800000:  # NaN: quiet, sign kept
        return ((u >> 16) & 0x8000) | 0x7FC0
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF) >> 16


def hex_bits(values, width: int) -> str:
    return "".join(f"{v & ((1 << (width * 8)) - 1):0{width * 2}x}" for v in values)


def fnv64(values) -> int:
    h = 0xCBF29CE484222325
    for v in values:
        for b in (v & 0xFF, (v >> 8) & 0xFF):
            h = ((h ^ b) * 0x100000001B3) & U64
    return h


def norm_row(x_bits, w_bits, dim, eps_bits):
    """The reference's (1+w) RMSNorm, one fp32 operation at a time."""
    eps = bits_to_f32(eps_bits)
    ss = np.float32(0.0)
    for d in range(dim):
        v = bf16_bits_to_f32(x_bits[d])
        ss = np.float32(ss + np.float32(v * v))
    rstd = np.float32(
        np.float32(1.0) / np.sqrt(np.float32(np.float32(ss) / np.float32(dim) + eps))
    )
    return [
        f32_to_bf16_bits(
            np.float32(
                np.float32(bf16_bits_to_f32(x_bits[d]) * rstd)
                * np.float32(np.float32(1.0) + bf16_bits_to_f32(w_bits[d]))
            )
        )
        for d in range(dim)
    ]


def rope_row(xn, cos_sin, dim, rotary):
    """The reference's rotation of an already-normalized bf16 row."""
    half = rotary // 2
    out = [0] * dim
    for d in range(dim):
        if d >= rotary:
            out[d] = xn[d]
            continue
        i = d if d < half else d - half
        xv = bf16_bits_to_f32(xn[d])
        rot = (
            np.float32(-bf16_bits_to_f32(xn[d + half]))
            if d < half
            else bf16_bits_to_f32(xn[d - half])
        )
        t1 = f32_to_bf16_bits(np.float32(xv * bf16_bits_to_f32(cos_sin[i])))
        t2 = f32_to_bf16_bits(np.float32(rot * bf16_bits_to_f32(cos_sin[half + i])))
        out[d] = f32_to_bf16_bits(np.float32(bf16_bits_to_f32(t1) + bf16_bits_to_f32(t2)))
    return out


def logit_bits(q_bits, k_bits, head_dim: int) -> int:
    acc = np.float32(0.0)
    for d in range(len(q_bits)):
        acc = np.float32(
            acc + np.float32(bf16_bits_to_f32(q_bits[d]) * bf16_bits_to_f32(k_bits[d]))
        )
    return f32_bits(np.float32(acc * np.float32(head_dim**-0.5)))


def rope_class(obj) -> str:
    return type(obj).__name__


def case_positions(factor: float):
    """The positions a case reports: the recipe's, plus its own last one."""
    last = int(ORIGINAL * factor) - 1
    return sorted(set(POSITIONS) | ({last} if last > max(POSITIONS) else set()))


def build_case(name, factor, betas, attn_factor, mrope_cache_factor, tc, x, w):
    """One parameter combination's goldens, out of vLLM's own objects.

    The table, the mscale and the cos/sin rows are vLLM's (the rows straight
    from the bf16 cache buffer the model reads). The rotated rows and the
    logit are this file's contract applied to those rows — see the module
    docstring: they gate the engine's rotation against vLLM's numbers.
    """
    from vllm.model_executor.layers.rotary_embedding.common import (
        yarn_find_correction_range,
    )

    rp = dict(tc.get("rope_parameters") or {})
    if mrope_cache_factor != 4.0:
        # No mrope_section in the dict and get_rope builds a plain
        # YaRNScalingRotaryEmbedding, whose own max_position_embeddings is
        # the original: mrope_cache_factor 1, the native band.
        rp.pop("mrope_section", None)
        rp.pop("mrope_interleaved", None)
    rp.update(
        {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": ORIGINAL,
            "beta_fast": betas[0],
            "beta_slow": betas[1],
            "attn_factor": attn_factor,
        }
    )
    rope = get_rope(
        head_size=tc["head_dim"],
        max_position=tc["max_position_embeddings"],
        rope_parameters=rp,
        dtype=torch.bfloat16,
    )
    dim, rotary = int(tc["head_dim"]), int(rope.rotary_dim)
    eps_bits = f32_bits(np.float32(tc["rms_norm_eps"]))
    mscale_bits = f32_bits(rope.mscale)
    band_from = int(rope.max_position_embeddings)
    low, high = yarn_find_correction_range(
        betas[0], betas[1], rotary, float(rope.base), band_from, True
    )
    inv = rope._compute_inv_freq(rope.scaling_factor)
    cache = rope.cos_sin_cache  # bf16: exactly what the model reads
    positions, n_cache = case_positions(factor), int(cache.shape[0])
    # The norm has no positional term: normalize once, rotate per position.
    xn = [norm_row(x[r], w[r], dim, eps_bits) for r in range(ROWS)]
    rows = []
    for pos in positions:
        assert pos < n_cache, f"{name}: position {pos} past the cache"
        cos_sin = [int(v) & 0xFFFF for v in cache[pos].view(torch.int16).tolist()]
        rot = [rope_row(xn[r], cos_sin, dim, rotary) for r in range(ROWS)]
        rows.append(
            {
                "pos": pos,
                "cache_row": hex_bits(cos_sin, 2),
                "q": hex_bits(rot[0], 2),
                "k": hex_bits(rot[1], 2),
                "logit": f"0x{logit_bits(rot[0], rot[1], dim):08x}",
            }
        )
    case = {
        "name": name,
        "params": {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": ORIGINAL,
            "beta_fast": betas[0],
            "beta_slow": betas[1],
            "attn_factor": attn_factor,
            "mrope_cache_factor": mrope_cache_factor,
        },
        "band": {
            "class": rope_class(rope),
            "correction_max_position": band_from,
            "low": float(low),
            "high": float(high),
            "cache_positions": n_cache,
        },
        "mscale_bits": f"0x{mscale_bits:08x}",
        "inv_freq": hex_bits([f32_bits(v) for v in inv.tolist()], 4),
        "rows": rows,
    }
    del rope, cache, inv, xn  # a 4x cache is 512 MB of bf16; do not pile them up
    return case


def build_fixture(checkpoint: str, image_digest: str) -> dict:
    from vllm.config import VllmConfig, set_current_vllm_config
    import vllm
    import vllm.model_executor.layers.rotary_embedding as rope_registry

    raw = json.load(open(f"{checkpoint}/config.json"))
    tc = raw["text_config"]
    dim = int(tc["head_dim"])
    rotary = int(dim * (tc.get("rope_parameters") or {}).get("partial_rotary_factor", 1.0))
    x, w = fixture_input(SEED, ROWS, dim)
    fx = {
        "format": FORMAT,
        "tool": "tests/python/qwen_yarn_oracle.py",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "container": {
            "image": IMAGE,
            "image_digest": image_digest,
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python": sys.version.split()[0],
        },
        "checkpoint": {
            "path": checkpoint,
            "head_dim": dim,
            "rotary_dim": rotary,
            "theta": (tc.get("rope_parameters") or {}).get("rope_theta", 10000),
            "rms_norm_eps_bits": f"0x{f32_bits(np.float32(tc['rms_norm_eps'])):08x}",
            "max_position_embeddings": int(tc["max_position_embeddings"]),
            "mrope_section": (tc.get("rope_parameters") or {}).get("mrope_section"),
        },
        "spec": {
            "seed": f"0x{SEED:016x}",
            "rows": ROWS,
            "prng": "splitmix64: s += 0x9e3779b97f4a7c15; z=s; z=(z^z>>30)*0xbf58476d1ce4e5b9;"
                     " z=(z^z>>27)*0x94d049bb133111eb; return z^z>>31",
            "lane": "x: sign r>>63, exponent 124 + r%5, mantissa (r>>11) & 0x7f; row-major,"
                    " one draw per lane",
            "weight": "w: sign r>>63, exponent 125, mantissa (r>>11) & 0x7f; drawn after every"
                      " x lane",
            "norm": "ss += v*v in lane order; rstd = 1/sqrt(ss/dim + eps);"
                    " xn = bf16((v*rstd)*(1+w))",
            "rotate": "neox half-split; out = bf16(bf16(xn*c) + bf16(rot*s)) with c/s the bf16"
                      " cache row; lanes at or past rotary_dim pass through",
            "logit": "fp32 sum_d(q[d]*k[d]) in lane order, times head_dim**-0.5",
            "fp": "every intermediate is fp32, rounded per operation, nothing fused",
            "input_fnv": f"0x{fnv64([b for row in x for b in row] + [b for row in w for b in row]):016x}",
        },
        "cases": [],
    }
    with set_current_vllm_config(VllmConfig()):
        for name, factor, betas, attn, mcf in CASES:
            rope_registry._ROPE_DICT.clear()  # get_rope memoizes; every case gets its own
            fx["cases"].append(build_case(name, factor, betas, attn, mcf, tc, x, w))
    return fx


def diff_fixture(got: dict, want: dict) -> list:
    """Every recorded value the recomputation disagrees with. Provenance
    excepted: a refresh always rewrites generated_utc (and the mount point
    is whoever ran it, so `path` is not a value)."""
    out = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k in ("generated_utc",) or path + "." + k == ".checkpoint.path":
                    continue
                if k not in a or k not in b:
                    out.append(f"{path}.{k}: present on one side only")
                else:
                    walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(f"{path}: length {len(b)} recorded, {len(a)} recomputed")
            for i, (p, q) in enumerate(zip(a, b)):
                walk(p, q, f"{path}[{i}]")
        elif a != b:
            out.append(f"{path}: recorded {b!r}, recomputed {a!r}")

    walk(got, want, "")
    return out


def fixture_mode(args, digest: str) -> int:
    if not os.path.isdir(args.checkpoint):
        print(
            f"{args.checkpoint}: not a snapshot directory — the fixture modes want\n"
            "  <hub>/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/<sha> mounted at /ckpt\n"
            "(see the docker line in this file's docstring)",
            file=sys.stderr,
        )
        return 2
    fx = build_fixture(args.checkpoint, digest)
    if args.emit:
        with open(args.emit, "w") as f:
            json.dump(fx, f, indent=1, sort_keys=True)
            f.write("\n")
        n = sum(len(c["rows"]) for c in fx["cases"])
        print(
            f"{args.emit}: {len(fx['cases'])} cases, {n} positions, "
            f"container {fx['container']['image_digest'][:20]}"
        )
        return 0
    want = json.load(open(args.check))
    if fx["container"]["image_digest"] == "unrecorded":
        # The values still came out of vLLM's yarn code, which is what the
        # comparison is about; only the attribution is missing, and the C++
        # gate is where that is enforced.
        print("note: no container digest passed, so the recorded one is not compared", file=sys.stderr)
        fx["container"]["image_digest"] = want["container"]["image_digest"]
    bad = diff_fixture(fx, want)
    if bad:
        print(
            f"{args.check}: {len(bad)} mismatch(es) against vLLM "
            f"{fx['container']['vllm_version']}:",
            file=sys.stderr,
        )
        for line in bad[:40]:
            print("  " + line, file=sys.stderr)
        return 1
    print(
        f"{args.check}: {len(fx['cases'])} cases, "
        f"{sum(len(c['rows']) for c in fx['cases'])} positions agree with vLLM "
        f"{fx['container']['vllm_version']} bit for bit"
    )
    return 0


def print_mode(checkpoint: str) -> int:
    from vllm.config import VllmConfig, set_current_vllm_config

    raw = json.load(open(f"{checkpoint}/config.json"))
    tc = raw["text_config"]
    merged = dict(tc.get("rope_parameters") or {})
    before = dict(merged)
    merged.update(
        {"rope_type": "yarn", "factor": FACTOR, "original_max_position_embeddings": ORIGINAL}
    )
    print("checkpoint rope_parameters:", before)
    print("after the launcher's merge:  ", merged)
    assert merged.get("mrope_section"), "the merge keeps mrope_section (MRotaryEmbedding)"

    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(head_size=tc["head_dim"], max_position=tc["max_position_embeddings"],
                        rope_parameters=merged, dtype=torch.bfloat16)
    print(f"class {rope_class(rope)} rotary_dim {rope.rotary_dim} base {float(rope.base)}")
    print(f"max_position_embeddings (the band's source) {rope.max_position_embeddings}")
    print(f"mscale {rope.mscale!r} f32 {hex(f32_bits(rope.mscale))}")
    inv = rope._compute_inv_freq(rope.scaling_factor)  # what the cache uses
    print("YaRN table (recipe, band 4x):", hexes(inv))
    cache = rope.cos_sin_cache
    for pos in POSITIONS:
        row = cache[pos]
        cos = ", ".join(repr(float(v)) for v in row[:4])
        sin = ", ".join(repr(float(v)) for v in row[32:36])
        print(f"  bf16 cache pos {pos}: cos {cos} ; sin {sin}")

    native = {k: v for k, v in merged.items() if k not in ("mrope_section", "mrope_interleaved")}
    with set_current_vllm_config(VllmConfig()):
        plain = get_rope(head_size=tc["head_dim"], max_position=ORIGINAL, rope_parameters=native,
                         dtype=torch.bfloat16)
    print(f"class {rope_class(plain)} (no mrope_section) mscale {plain.mscale!r}")
    print("YaRN table (native band 1x):", hexes(plain._compute_inv_freq(plain.scaling_factor)))
    return 0


def hexes(t: torch.Tensor) -> str:
    return "[" + ", ".join(hex(int(x)) for x in t.view(torch.int32)) + "]"


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="The Qwen YaRN goldens and their fixtures.")
    ap.add_argument("checkpoint", nargs="?", default="/ckpt",
                    help="the snapshot directory (default /ckpt, the container's mount)")
    ap.add_argument("--emit", metavar="PATH", help="write the machine-readable fixture there")
    ap.add_argument("--check", metavar="PATH", help="recompute a fixture; exit non-zero on drift")
    ap.add_argument("--image-digest", default=os.environ.get("DGPP_ORACLE_IMAGE_DIGEST", ""),
                    help="sha256:... of the pinned container (or $DGPP_ORACLE_IMAGE_DIGEST)")
    args = ap.parse_args(argv)
    digest = args.image_digest or "unrecorded"
    if args.emit or args.check:
        if digest == "unrecorded":
            print(
                "WARNING: no container digest (--image-digest / $DGPP_ORACLE_IMAGE_DIGEST): an "
                "--emit file will carry \"unrecorded\" and the C++ fixture gate will refuse it",
                file=sys.stderr,
            )
        return fixture_mode(args, digest)
    return print_mode(args.checkpoint)


if __name__ == "__main__":
    from vllm.model_executor.layers.rotary_embedding import get_rope

    raise SystemExit(main(sys.argv[1:]))
