#!/usr/bin/env python3
"""Inkling 2 vision tower reference (docs/inkling2_mini_plan.md section 1.8).

Computes the tower in numpy from the SAME checkpoint the engine loads and
writes the comparable stages as a DGPPINK2 dump read by
tests/cuda/inkling2_vision_test.cpp. It also writes the tiny tower the tests
run on, so no test needs the release.

The tower: an image padded right and bottom to 40-pixel cells is cut into
native 20-pixel patches in raster order. Each patch goes through a linear
embedding without bias and takes a position row resampled bilinearly from
the learned grid. A class row leads the sequence. After a LayerNorm come the
pre-norm layers (full attention inside the image with an axial rotation, a
GELU MLP, a learned scale on each sublayer output), then the four patch rows
of every cell are concatenated and projected to the text width. No norm
follows the last layer.

Rounding follows the released kernels (vLLM's models/inkling2/nvidia/ops/
vision.py): a value is rounded to bf16 where the reference's eager ops store
a tensor, and the arithmetic in between is float32 where the kernels spell
it out operation by operation (the pixel map, the bilinear weights, the
rotation) and float64 here for the reductions.

  pixels            2 * (bf16((v / 255 - mean) / clip_std) * std + mean) - 1, bf16
                    v = -1 right of and below the image's content
  patch embedding   bf16(dot)
  position          bf16(lerp(lerp(top), lerp(bottom))), align_corners false
  input rows        bf16(patch + position), bf16(class + class position)
  LayerNorm         bf16(w * (x - mean) * rstd + b)
  projections       bf16(dot + bias)
  rotation          bf16 of each rotated pair
  attention         wide softmax, bf16(output)
  residual          bf16(x + bf16(update * gamma))
  GELU              bf16(x * 0.5 * (1 + erf(x / sqrt(2))))

Usage:
  inkling2_vision_reference.py write-fixture --out DIR
  inkling2_vision_reference.py gen --checkpoint-dir DIR --case rect|padded|minimum|tiled --out FILE
  inkling2_vision_reference.py gen --checkpoint-dir DIR --width W --height H [--seed S] --out FILE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_reference_dump import bf16  # noqa: E402
from glm_reference_dump import _parse_shard_header  # noqa: E402

MAGIC = b"DGPPINK2"
VERSION = 1
PREFIX = "model.visual."
GRID = 40

# The processor's CLIP statistics as the released kernel takes them: float32
# scalars, the deviations once as the processor rounds them and once exact.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
PROCESSOR_STD = np.array([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# The tiny tower of the tests: the text width of the tests' tiny text model,
# four heads of sixteen (four rotation frequencies an axis) and a 6 x 6
# position grid, so every image of the tests resamples it.
TINY = {
    "model_type": "inkling2_vision", "text_hidden_size": 256, "temporal_patch_size": 2,
    "native_patch_size": 20, "adapter_merge_size": 2, "projection_hidden_dim": 96,
    "projector_bias": False, "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "mlp_ratio": 4.0, "posemb_grid_size": 6, "layer_norm_eps": 1e-05,
}

CASES = {
    # name: (content width, content height, seed)
    "rect": (120, 80, 3),       # 3 x 2 cells, no padding
    "padded": (93, 57, 5),      # padded to 120 x 80
    "minimum": (1, 1, 7),       # one cell, almost all padding
    "tiled": (320, 320, 11),    # 257 rows: two attention tiles
}


def f32(x):
    return np.asarray(x, dtype=np.float32)


def bf16_f32(x):
    """Round float32 values to bf16, as float32."""
    return bf16(x).astype(np.float32)


def bf16_bytes(values):
    a = np.ascontiguousarray(bf16(values).astype(np.float32))
    return (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def erf(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size > 2_000_000:
        try:
            import torch
            return torch.erf(torch.from_numpy(x)).numpy()
        except ImportError:
            pass
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def gelu(x):
    return x * 0.5 * (1.0 + erf(x * 0.7071067811865476))


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

def read_index(model_dir):
    """Every visual tensor: name without the prefix -> (path, offset, nbytes,
    dtype, shape)."""
    entries = {}
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors") and name != "mtp.safetensors":
            _parse_shard_header(os.path.join(model_dir, name), entries)
    visual = {k[len(PREFIX):]: v for k, v in entries.items() if k.startswith(PREFIX)}
    if not visual:
        raise FileNotFoundError("no model.visual tensors in %s" % model_dir)
    return visual


def load(entries, name):
    path, offset, nbytes, dtype, shape = entries[name]
    if dtype != "BF16":
        raise ValueError("%s is %s, expected BF16" % (name, dtype))
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(nbytes)
    w = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return w.view(np.float32).astype(np.float64).reshape(shape)


def vision_config(model_dir):
    path = os.path.join(model_dir, "vision_config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)["vision_config"]


# ---------------------------------------------------------------------------
# the tower
# ---------------------------------------------------------------------------

def pixel_patches(rgb, valid_w, valid_h, patch):
    """The canvas [H, W, 3] uint8 as native patch rows [hp * wp, 3 * patch *
    patch] in raster order, features (py, px, channel). Every operation is a
    float32 operation rounded on its own, as the kernel's."""
    h, w, _ = rgb.shape
    v = rgb.astype(np.float32)
    v[valid_h:, :, :] = np.float32(-1.0)
    v[:, valid_w:, :] = np.float32(-1.0)
    clip = bf16_f32((v / np.float32(255.0) - CLIP_MEAN) / PROCESSOR_STD)
    out = bf16(np.float32(2.0) * (clip * CLIP_STD + CLIP_MEAN) - np.float32(1.0))
    hp, wp = h // patch, w // patch
    rows = out.reshape(hp, patch, wp, patch, 3).transpose(0, 2, 1, 3, 4)
    return rows.reshape(hp * wp, patch * patch * 3)


def axis_taps(index, target, source):
    """align_corners false source taps of `index` in [0, target), clamped to
    [0, source - 1], in float32."""
    coord = np.float32(source) * (index.astype(np.float32) + np.float32(0.5)) / np.float32(target) - np.float32(0.5)
    coord = np.maximum(np.float32(0.0), np.minimum(coord, np.float32(source - 1)))
    lo = np.floor(coord).astype(np.int64)
    hi = np.minimum(lo + 1, source - 1)
    w_hi = coord - lo.astype(np.float32)
    w_lo = np.float32(1.0) - w_hi
    return lo, hi, w_lo, w_hi


def lerp(a, b, wa, wb):
    """The kernel's fused form: the larger weight leads the multiply-add.
    float32 values, the product kept exact in float64 before the one
    rounding."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    wa, wb = wa.astype(np.float64), wb.astype(np.float64)
    first = a * wa + f32(b * wb).astype(np.float64)
    second = b * wb + f32(a * wa).astype(np.float64)
    return f32(np.where(wa >= wb, first, second))


def position_rows(table, hp, wp, side):
    """bf16 position rows [hp * wp, H] of the [side, side, H] table."""
    ys, xs = np.divmod(np.arange(hp * wp), wp)
    y0, y1, wy0, wy1 = axis_taps(ys, hp, side)
    x0, x1, wx0, wx1 = axis_taps(xs, wp, side)
    grid = table.reshape(side, side, -1).astype(np.float32)
    top = lerp(grid[y0, x0], grid[y0, x1], wx0[:, None], wx1[:, None])
    bottom = lerp(grid[y1, x0], grid[y1, x1], wx0[:, None], wx1[:, None])
    return bf16_f32(lerp(top, bottom, wy0[:, None], wy1[:, None]))


def layer_norm(x, w, b, eps):
    mean = x.mean(axis=-1, keepdims=True)
    diff = x - mean
    rstd = 1.0 / np.sqrt((diff * diff).mean(axis=-1, keepdims=True) + eps)
    return bf16(w * (diff * rstd) + b)


def rotate(qk, wp, heads, dim):
    """qk [n, heads * dim] bf16 -> the rotated bf16 rows. Pairs (2p, 2p + 1):
    the first dim / 4 pairs turn by column + 1, the rest by row + 1. Row 0 is
    the class row at 0 on both axes. float32 angles and products."""
    n = qk.shape[0]
    quarter = dim // 4
    inv_freq = f32([10000.0 ** (-d / dim) for d in range(0, dim, 4)])
    ys, xs = np.divmod(np.arange(n - 1), wp)
    col = np.concatenate([[0], xs + 1]).astype(np.float32)
    row = np.concatenate([[0], ys + 1]).astype(np.float32)
    angle = np.concatenate([col[:, None] * inv_freq[None, :], row[:, None] * inv_freq[None, :]], axis=1)  # [n, dim/2]
    assert angle.shape == (n, 2 * quarter)
    cos, sin = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    x = qk.astype(np.float32).reshape(n, heads, dim // 2, 2)
    even, odd = x[..., 0].astype(np.float64), x[..., 1].astype(np.float64)
    c, s = cos[:, None, :].astype(np.float64), sin[:, None, :].astype(np.float64)
    # fma(even, cos, -(odd * sin)) and fma(odd, cos, even * sin)
    new_even = f32(even * c - f32(odd * s).astype(np.float64))
    new_odd = f32(odd * c + f32(even * s).astype(np.float64))
    return bf16(np.stack([new_even, new_odd], axis=-1).reshape(n, heads * dim))


def encode(cfg, entries, rgb, valid_w, valid_h, progress=False):
    """The stages of one image: a dict of float64 arrays holding bf16 values."""
    patch, merge = cfg["native_patch_size"], cfg["adapter_merge_size"]
    H, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    dim, side, eps = H // heads, cfg["posemb_grid_size"], cfg["layer_norm_eps"]
    h, w, _ = rgb.shape
    if patch * merge != GRID or h % GRID or w % GRID:
        raise ValueError("the canvas is not on the 40-pixel grid")
    hp, wp = h // patch, w // patch
    n = hp * wp + 1
    b = "backbone."
    stages = {}
    patches = pixel_patches(rgb, valid_w, valid_h, patch)
    stages["patches"] = patches
    patch_embed = bf16(patches @ load(entries, b + "embeddings.patch_embedding.weight").T)
    stages["patch_embed"] = patch_embed
    position = position_rows(load(entries, b + "embeddings.position_embedding"), hp, wp, side)
    rows = np.empty((n, H))
    rows[0] = bf16(f32(load(entries, b + "embeddings.class_embedding")) +
                   f32(load(entries, b + "embeddings.cls_position_embedding")))
    rows[1:] = bf16(f32(patch_embed) + position)
    stages["embed"] = rows.copy()
    x = layer_norm(rows, load(entries, b + "ln_pre.weight"), load(entries, b + "ln_pre.bias"), eps)
    stages["ln_pre"] = x.copy()
    scale = dim ** -0.5
    for l in range(cfg["num_hidden_layers"]):
        p = b + "layers.%d." % l
        if progress:
            print("layer %d" % l, file=sys.stderr, flush=True)
        normed = layer_norm(x, load(entries, p + "ln_1.weight"), load(entries, p + "ln_1.bias"), eps)
        proj = {}
        for name in ("q", "k", "v"):
            weight = load(entries, p + "self_attn.%s_proj.weight" % name)
            bias = load(entries, p + "self_attn.%s_proj.bias" % name)
            proj[name] = bf16(normed @ weight.T + bias)
        q = rotate(proj["q"], wp, heads, dim).reshape(n, heads, dim)
        k = rotate(proj["k"], wp, heads, dim).reshape(n, heads, dim)
        v = proj["v"].reshape(n, heads, dim)
        attn = np.empty((n, heads, dim))
        for head in range(heads):
            logits = (q[:, head] @ k[:, head].T) * scale
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            attn[:, head] = prob @ v[:, head]
        attn = bf16(attn.reshape(n, H))
        out = bf16(attn @ load(entries, p + "self_attn.out_proj.weight").T + load(entries, p + "self_attn.out_proj.bias"))
        x = bf16(f32(x) + bf16_f32(f32(out) * f32(load(entries, p + "ls_1.gamma"))))
        normed = layer_norm(x, load(entries, p + "ln_2.weight"), load(entries, p + "ln_2.bias"), eps)
        hidden = bf16(normed @ load(entries, p + "mlp.c_fc.weight").T + load(entries, p + "mlp.c_fc.bias"))
        hidden = bf16(gelu(hidden))
        out = bf16(hidden @ load(entries, p + "mlp.c_proj.weight").T + load(entries, p + "mlp.c_proj.bias"))
        x = bf16(f32(x) + bf16_f32(f32(out) * f32(load(entries, p + "ls_2.gamma"))))
        stages["layer%d" % l] = x.copy()
    # a cell's patches in (dy, dx) order, side by side
    grid = x[1:].reshape(hp // merge, merge, wp // merge, merge, H).transpose(0, 2, 1, 3, 4)
    pooled = grid.reshape((hp // merge) * (wp // merge), merge * merge * H)
    stages["pooled"] = pooled
    project = bf16(gelu(bf16(pooled @ load(entries, "proj_in.weight").T)))
    stages["image_rows"] = bf16(project @ load(entries, "proj_out.weight").T)
    return stages


# ---------------------------------------------------------------------------
# the tiny fixture, the test images and the dump
# ---------------------------------------------------------------------------

def expected_tensors(cfg):
    """Checkpoint names and shapes (src/models/inkling2/binding.cpp), each
    with the magnitude the fixture draws it at."""
    H, patch, merge = cfg["hidden_size"], cfg["native_patch_size"], cfg["adapter_merge_size"]
    mlp = int(H * cfg["mlp_ratio"])
    side = cfg["posemb_grid_size"]
    b = PREFIX + "backbone."
    out = [
        (b + "embeddings.class_embedding", [H], ("normal", 0.5)),
        (b + "embeddings.cls_position_embedding", [H], ("normal", 0.5)),
        (b + "embeddings.patch_embedding.weight", [H, 3 * patch * patch], ("normal", 0.04)),
        (b + "embeddings.position_embedding", [side * side, H], ("normal", 0.5)),
        (b + "ln_pre.weight", [H], ("gain", 0.2)),
        (b + "ln_pre.bias", [H], ("normal", 0.1)),
    ]
    for l in range(cfg["num_hidden_layers"]):
        p = b + "layers.%d." % l
        for name in ("ln_1", "ln_2"):
            out.append((p + name + ".weight", [H], ("gain", 0.2)))
            out.append((p + name + ".bias", [H], ("normal", 0.1)))
        out.append((p + "ls_1.gamma", [H], ("normal", 0.5)))
        out.append((p + "ls_2.gamma", [H], ("normal", 0.5)))
        for name, std in (("q_proj", 0.3), ("k_proj", 0.3), ("v_proj", 0.15), ("out_proj", 0.15)):
            out.append((p + "self_attn.%s.weight" % name, [H, H], ("normal", std)))
            out.append((p + "self_attn.%s.bias" % name, [H], ("normal", 0.05)))
        out.append((p + "mlp.c_fc.weight", [mlp, H], ("normal", 0.15)))
        out.append((p + "mlp.c_fc.bias", [mlp], ("normal", 0.05)))
        out.append((p + "mlp.c_proj.weight", [H, mlp], ("normal", 0.08)))
        out.append((p + "mlp.c_proj.bias", [H], ("normal", 0.05)))
    out.append((PREFIX + "proj_in.weight", [cfg["projection_hidden_dim"], merge * merge * H], ("normal", 0.08)))
    out.append((PREFIX + "proj_out.weight", [cfg["text_hidden_size"], cfg["projection_hidden_dim"]], ("normal", 0.15)))
    return out


def write_fixture(args):
    """A tiny tower in the release's naming: vision_config.json, a config.json
    that holds it, and model.safetensors with the visual tensors only. The
    magnitudes keep the attention selective, so the rotation and the position
    rows matter to the output."""
    cfg = dict(TINY)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    header, payload = {}, bytearray()
    for name, shape, (kind, size) in expected_tensors(cfg):
        values = rng.standard_normal(shape) * size
        if kind == "gain":
            values = 1.0 + values
        blob = bf16_bytes(values.reshape(-1))
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [len(payload), len(payload) + len(blob)]}
        payload.extend(blob)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(os.path.join(args.out, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(bytes(payload))
    with open(os.path.join(args.out, "vision_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"vision_config": cfg}, f, indent=1)
    print("wrote the tiny tower to %s (%d tensors, %d bytes)" % (args.out, len(header), len(payload)))
    return 0


def test_image(width, height, seed):
    """Deterministic content [height, width, 3]: gradients with an 8-pixel
    checker, so neighbouring patches and both axes differ."""
    y, x = np.mgrid[0:height, 0:width]
    channels = [(x * 7 + y * 13 + c * 29 + seed * 31 + ((x // 8 + y // 8) % 2) * 40) % 256 for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def canvas_of(content):
    """The content on a canvas padded right and bottom to the grid. The
    padding bytes are 0: the tower does not read them."""
    h, w, _ = content.shape
    ch, cw = -(-h // GRID) * GRID, -(-w // GRID) * GRID
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    canvas[:h, :w] = content
    return canvas


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-inkling2-vision-reference-dump", "version": VERSION, "backend": "numpy",
              "config": cfg_summary, "tensors": {}}
    payload = bytearray()
    for name, (dtype, shape, blob) in tensors.items():
        header["tensors"][name] = {"dtype": dtype, "shape": list(shape), "offset": len(payload), "nbytes": len(blob)}
        payload.extend(blob)
    hb = json.dumps(header, indent=1, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(hb)))
        f.write(hb)
        f.write(bytes(payload))


def gen(args):
    cfg = vision_config(args.checkpoint_dir)
    entries = read_index(args.checkpoint_dir)
    if args.case:
        width, height, seed = CASES[args.case]
    else:
        if not args.width or not args.height:
            raise SystemExit("gen: give --case or --width and --height")
        width, height, seed = args.width, args.height, args.seed
    content = test_image(width, height, seed)
    canvas = canvas_of(content)
    ch, cw, _ = canvas.shape
    tokens = (ch // GRID) * (cw // GRID)
    # 0 x 0 names a canvas that is all content, as the engine's ImageInput.
    padded = (cw, ch) != (width, height)
    stages = encode(cfg, entries, canvas, width, height, progress=True)
    tensors = {
        "geometry": ("I32", [6], struct.pack("<6i", cw, ch, width if padded else 0, height if padded else 0, GRID, tokens)),
        "rgb": ("U8", [ch * cw * 3], canvas.tobytes()),
    }
    for name, value in stages.items():
        tensors[name] = ("BF16", list(value.shape), bf16_bytes(value.reshape(-1)))
    summary = {"hidden": cfg["text_hidden_size"], "vocab": 0, "visits": 0, "top_k": 0, "tokens": tokens,
               "activations": "vision", "layers": cfg["num_hidden_layers"], "vision_hidden": cfg["hidden_size"]}
    write_dump(args.out, summary, tensors)
    print("wrote %s (%d x %d content on a %d x %d canvas, %d tokens)" % (args.out, width, height, cw, ch, tokens))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-fixture")
    w.add_argument("--out", required=True)
    w.add_argument("--seed", type=int, default=20261122)
    g = sub.add_parser("gen")
    g.add_argument("--checkpoint-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--case", choices=sorted(CASES))
    g.add_argument("--width", type=int, default=0)
    g.add_argument("--height", type=int, default=0)
    g.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    return write_fixture(args) if args.cmd == "write-fixture" else gen(args)


if __name__ == "__main__":
    sys.exit(main())
