#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash reference dump (docs/deepseek_v41_flash_plan.md G5,
2026-09-13).

Computes the forward over a prompt — the embedding on the four residual
streams, the single-pass mHC (inference/model.py Block: every sublayer
collapses with the PREVIOUS sublayer's pre), the Engram layers (the
hashed n-gram rows out of the tables, the joint key / value projection,
the signed-sqrt sigmoid gate), the CSA2 attention (the window over the
fp8_block rows, the compressor at ratio 1 / 2, the index keys, the fp4
e8m0/32 indexer, the candidate pool, the top-k over the compressed
entries, the two-source attention with the sink, the inverse rotation,
the grouped wo_a), the sqrtsoftplus-routed MoE with its MXFP4 experts and
fp8 shared expert, the head's weighted collapse — in numpy doubles with
bf16 rounding at the ENGINE's boundaries (the exact weight decodes, bf16
projection outputs, the fp32 mHC exports, the one-rounding norms and
updates, the release's activation quantizers for the caches), from the
SAME fixture the engine runs, and writes the comparable outputs (tokens,
every layer's streams, the final read, per-token top-k logits, the
routing decisions with their boundary margins, every index source's
selection with its margin) as a DGPPGDSD dump read by
tests/cuda/dsv41_forward_test.cpp.

Usage:
  dsv41_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE [--tokens T] [--seed S]
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
from glm_reference_dump import read_safetensors_index  # noqa: E402

MAGIC = b"DGPPGDSD"
VERSION = 1
MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# rounding
# ---------------------------------------------------------------------------
def bf16(x):
    """Round-to-nearest-even to bf16, returned as float64 (exact values)."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    b = a.view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def e4m3_round(x):
    """The nearest e4m3 value (RNE, saturating at 448; subnormals at 2^-9)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    s = np.sign(np.asarray(x, dtype=np.float64))
    out = np.empty_like(a)
    sub = a < 2.0 ** -6
    out[sub] = np.rint(a[sub] * 512.0) * 2.0 ** -9
    m, e = np.frexp(a[~sub])  # a = m * 2^e, m in [0.5, 1)
    q = np.rint((m * 2.0 - 1.0) * 8.0)
    out[~sub] = np.minimum((1.0 + q / 8.0) * np.exp2(e - 1), 448.0)
    return s * out


def e2m1_round(x):
    """The nearest e2m1 value on {0, .5, 1, 1.5, 2, 3, 4, 6} (ties to the even code)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    s = np.sign(np.asarray(x, dtype=np.float64))
    out = np.select([a <= 0.25, a < 0.75, a <= 1.25, a < 1.75, a <= 2.5, a < 3.5, a <= 5.0],
                    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], 6.0)
    return s * out


def e8m0_ceil_pow2(x):
    """2^ceil(log2(x)) from the fp32 bit fields (kernel.py fast_round_scale)."""
    xf = np.asarray(x, dtype=np.float32)
    bits = xf.view(np.uint32).astype(np.int64)
    k = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0)
    return np.exp2(k.astype(np.float64))


def e4m3_decode(codes):
    c = np.asarray(codes, dtype=np.uint8).astype(np.int64)
    sign = np.where(c & 0x80, -1.0, 1.0)
    e = (c >> 3) & 0xF
    m = c & 0x7
    val = np.where(e == 0, m * 2.0 ** -9, (1.0 + m / 8.0) * np.exp2(e.astype(np.float64) - 7.0))
    nan = (e == 15) & (m == 7)
    val = np.where(nan, np.nan, val)
    return sign * val


E2M1_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e8m0_decode(bytes_):
    b = np.asarray(bytes_, dtype=np.uint8).astype(np.float64)
    return np.exp2(b - 127.0)


# ---------------------------------------------------------------------------
# the checkpoint
# ---------------------------------------------------------------------------
def load_np(entries, name):
    path, offset, nbytes, dtype, shape = entries[name]
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(nbytes)
    if dtype == "BF16":
        w = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return w.view(np.float32).astype(np.float64).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).astype(np.float64).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(buf, dtype=np.float16).astype(np.float64).reshape(shape)
    if dtype == "I16":
        return np.frombuffer(buf, dtype=np.int16).reshape(shape)
    if dtype in ("F8_E4M3", "F8_E8M0", "I8"):
        return np.frombuffer(buf, dtype=np.uint8).reshape(shape)
    raise ValueError("unsupported dtype %s for %s" % (dtype, name))


# ---- the EXL3 trellis (docs/deepseek_v41_flash_exl3_plan.md §2; models/exl3_matrix.hpp) ----
# A matrix is `trellis` I16 [K/16, N/16, 16 * bits] (a 16 x 16 tile per
# (k tile, n tile), its 256 values in the mma B-fragment order packed
# MSB-first into little-endian uint32 words), `suh` F16 [K], `svh` F16 [N].
# State p of a tile is the stream's bits [(p + 1) bits - 16, (p + 1) bits)
# modulo 256 bits; its value under the mul1 codebook is one fp16 fma of
# 1024 + bytesum(state * 0x83DCD12D) by 0x1eee, 0xc931. The weight in the
# HF layout ([N, K]) is svh o H_N (w_inner) H_K o suh with the 128-point
# Sylvester-order rotations over sqrt(128) per block, in doubles here.
_MUL1_TABLE = None


def mul1_table():
    global _MUL1_TABLE
    if _MUL1_TABLE is None:
        st = np.arange(65536, dtype=np.uint64)
        x = (st * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)
        bs = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        h = (0x6400 + bs).astype(np.uint16).view(np.float16).astype(np.float64)
        kinv = np.array([0x1eee], dtype=np.uint16).view(np.float16).astype(np.float64)[0]
        kbias = np.array([0xc931], dtype=np.uint16).view(np.float16).astype(np.float64)[0]
        _MUL1_TABLE = (h * kinv + kbias).astype(np.float16).astype(np.float64)  # one rounding, as the fma
    return _MUL1_TABLE


def _tile_positions():
    p = np.arange(256)
    t, j = p // 8, p % 8
    rows = (t % 4) * 2 + np.array([0, 1, 8, 9])[j % 4]
    cols = t // 4 + 8 * (j // 4)
    return rows, cols


def exl3_inner(trellis):
    """The trellis [K/16, N/16, 16 * bits] int16 -> w_inner [K, N] (the codebook values)."""
    kt, nt, words16 = trellis.shape
    bits = words16 // 16
    length = 256 * bits
    # Every tile's bit stream: words LE uint32, MSB-first inside each word.
    u32 = np.ascontiguousarray(trellis).view(np.uint8).reshape(kt, nt, length // 32, 4)
    u32 = (u32[..., 0].astype(np.uint32) | (u32[..., 1].astype(np.uint32) << 8) |
           (u32[..., 2].astype(np.uint32) << 16) | (u32[..., 3].astype(np.uint32) << 24))
    bitpos = np.arange(length)
    stream = ((u32[:, :, bitpos // 32] >> (31 - (bitpos % 32)).astype(np.uint32)) & 1).astype(np.uint32)  # [kt, nt, length]
    p = np.arange(256)
    end = ((p + 1) * bits) % length
    idx = (end[:, None] - 16 + np.arange(16)[None, :]) % length                        # [256, 16]
    bits16 = stream[:, :, idx]                                                           # [kt, nt, 256, 16]
    weights = (np.uint32(1) << np.arange(15, -1, -1, dtype=np.uint32))
    states = (bits16 * weights).sum(axis=-1).astype(np.int64)                            # [kt, nt, 256]
    vals = mul1_table()[states]
    rows, cols = _tile_positions()
    tile = np.zeros((kt, nt, 16, 16), dtype=np.float64)
    tile[:, :, rows, cols] = vals
    return tile.transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def hadamard128_blocks(x, axis):
    """The 128-point rotation (butterflies at strides 1..64, element j: v_j + v_{j^s} when
    bit s is clear, v_{j^s} - v_j when set) over sqrt(128), per 128-block along `axis`."""
    x = np.moveaxis(x, axis, -1)
    shape = x.shape
    v = x.reshape(shape[:-1] + (shape[-1] // 128, 128)).copy()
    j = np.arange(128)
    for s in (1, 2, 4, 8, 16, 32, 64):
        other = v[..., j ^ s]
        v = np.where((j & s) != 0, other - v, v + other)
    v = v * (1.0 / np.sqrt(128.0))
    return np.moveaxis(v.reshape(shape), -1, axis)


def hadamard128_blocks_f32(x):
    """The input rotation as the engine's rotate kernel computes it: fp32
    butterflies in the stage order 1..64 over each 128-block of the last
    axis, then one multiply by the fp32 scale 0.088388347648."""
    x = np.asarray(x, dtype=np.float32)
    shape = x.shape
    v = x.reshape(shape[:-1] + (shape[-1] // 128, 128)).copy()
    j = np.arange(128)
    for s_ in (1, 2, 4, 8, 16, 32, 64):
        other = v[..., j ^ s_]
        v = np.where((j & s_) != 0, other - v, v + other).astype(np.float32)
    v = (v * np.float32(0.088388347648)).astype(np.float32)
    return v.reshape(shape)


class Exl3Mat:
    """An EXL3 matrix in its inner form, so a product models the engine's
    chain (docs/deepseek_v41_flash_exl3_plan.md §2.2): the row is scaled
    by suh and rotated in fp32 and rounded to fp16 — the one rounding the
    engine adds, bitwise its rotate kernel's — then multiplied by the
    codebook values (exact products, fp32 accumulation in the engine),
    rotated on the output axis and scaled by svh in doubles. `x @ m.T`
    and `linear_bf16(x, m)` take it like a dense [N, K] array; a row slice
    on 128-blocks (`m[a:b]`) is the output block range."""
    __array_ufunc__ = None  # `ndarray @ Exl3T` defers to our __rmatmul__

    def __init__(self, inner, suh, svh):
        self.inner, self.suh, self.svh = inner, suh, svh  # inner [K, N], suh [K], svh [N]

    @property
    def shape(self):
        return (self.svh.shape[0], self.suh.shape[0])

    def __getitem__(self, rows):
        start, stop, step = rows.indices(self.shape[0])
        assert step == 1 and start % 128 == 0 and stop % 128 == 0, "an EXL3 row slice is whole 128-blocks"
        return Exl3Mat(self.inner[:, start:stop], self.suh, self.svh[start:stop])

    @property
    def T(self):
        return _Exl3T(self)

    def apply(self, x):
        """x [..., K] -> the unrounded product [..., N] (doubles)."""
        x = np.asarray(x, dtype=np.float64)
        xh = hadamard128_blocks_f32(x.astype(np.float32) * self.suh.astype(np.float32)).astype(np.float16).astype(np.float64)
        y = xh @ self.inner
        return hadamard128_blocks(y, axis=-1) * self.svh

    def dense(self):
        """The exact dequantized W^T ([N, K] doubles) for an oracle that wants the matrix."""
        w = hadamard128_blocks(self.inner, axis=0) * self.suh[:, None]
        return (hadamard128_blocks(w, axis=1) * self.svh[None, :]).T


class _Exl3T:
    __array_ufunc__ = None

    def __init__(self, m):
        self.m = m

    def __rmatmul__(self, x):
        return self.m.apply(x)


def load_exl3(entries, base):
    """An EXL3 matrix -> its Exl3Mat (the engine's chain modelled on a product)."""
    trellis = load_np(entries, base + ".trellis")
    suh = load_np(entries, base + ".suh")
    svh = load_np(entries, base + ".svh")
    return Exl3Mat(exl3_inner(trellis), suh, svh)


def head_weight(entries):
    """The lm head [V, H]: the EXL3 pack's matrix or the release's bf16 rows."""
    if "head.trellis" in entries:
        return load_exl3(entries, "head")
    return load_np(entries, "head.weight")


def load_w(entries, base):
    """A projection in whichever format the checkpoint packs: EXL3, else the fp8 pair."""
    if base + ".trellis" in entries:
        return load_exl3(entries, base)
    return load_fp8(entries, base)


def load_expert(entries, base):
    if base + ".trellis" in entries:
        return load_exl3(entries, base)
    return load_mxfp4(entries, base)


def load_fp8(entries, base):
    """An fp8 pair on the 32 x 32 grid -> the exact dequantized [N, K] floats."""
    p = load_np(entries, base + ".weight")
    s = load_np(entries, base + ".scale")
    n, k = p.shape
    sc = np.repeat(np.repeat(e8m0_decode(s), 32, axis=0), 32, axis=1)[:n, :k]
    return e4m3_decode(p) * sc


def load_mxfp4(entries, base):
    """An MXFP4 pair -> the exact [N, K] floats (low nibble = even element)."""
    p = load_np(entries, base + ".weight")
    s = load_np(entries, base + ".scale")
    n, half = p.shape
    codes = np.empty((n, half * 2), dtype=np.int64)
    codes[:, 0::2] = p & 0xF
    codes[:, 1::2] = p >> 4
    vals = E2M1_TABLE[codes]
    sc = np.repeat(e8m0_decode(s), 32, axis=1)[:, : half * 2]
    return vals * sc


def load_table_rows(entries, base, ids):
    """Engram table rows (e4m3 payload, e8m0 scales per 32) for `ids` -> bf16 values [n, 256]."""
    p = load_np(entries, base + ".weight")
    s = load_np(entries, base + ".scale")
    rows = e4m3_decode(p[ids]) * np.repeat(e8m0_decode(s[ids]), 32, axis=1)
    return bf16(rows)


def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        root = json.load(f)
    return root["text_config"], root


def load_sidecar(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "dgpp_engram_tables.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------
def rmsnorm1(x, w, eps):
    """The one-rounding RMSNorm over the last axis: bf16(w * (x * rsqrt(mean(x^2) + eps)))."""
    x = np.asarray(x, dtype=np.float64)
    rs = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return bf16(w * (x * rs))


def linear_bf16(x, w):
    """bf16(x @ w^T): a projection with a bf16 output (the engine's GEMV / tile kernels)."""
    if isinstance(w, Exl3Mat):
        return bf16(w.apply(x))
    return bf16(np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64).T)


def inv_freq(dim, theta, original, factor, beta_fast, beta_slow):
    freqs = np.array([1.0 / float(np.float32(theta) ** np.float32((2.0 * i) / dim)) for i in range(dim // 2)],
                     dtype=np.float32)
    freqs = np.array([np.float32(1.0) / np.float32(np.float32(theta) ** np.float32((2.0 * i) / dim)) for i in range(dim // 2)],
                     dtype=np.float32)
    if original > 0:
        def corrected(rot):
            return dim * math.log(original / (rot * 2 * math.pi)) / (2 * math.log(theta))
        low = max(math.floor(corrected(beta_fast)), 0)
        high = min(math.ceil(corrected(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(max(high - low, 1e-3)), 0, 1)
        smooth = np.float32(1.0) - ramp
        freqs = (freqs / np.float32(factor) * (np.float32(1.0) - smooth) + freqs * smooth).astype(np.float32)
    return freqs


def rope(x, pos, freq, inverse=False):
    """Rotate the last 64 of the last axis of x [.., 64+] at positions pos (one per row): one bf16 rounding."""
    x = np.array(x, dtype=np.float64)
    tail = x[..., -64:]
    ang = (np.asarray(pos, dtype=np.float32)[:, None] * freq[None, :]).astype(np.float64)  # [rows, 32]
    c, s = np.cos(ang), np.sin(ang)
    if inverse:
        s = -s
    while c.ndim < tail.ndim - 1 + 1:
        c = c[:, None, :]
        s = s[:, None, :]
    x0, x1 = tail[..., 0::2], tail[..., 1::2]
    r0 = bf16(x0 * c - x1 * s)
    r1 = bf16(x1 * c + x0 * s)
    out = tail.copy()
    out[..., 0::2] = r0
    out[..., 1::2] = r1
    x[..., -64:] = out
    return x


def quant_fp8_block(x):
    """act_quant(block 32, ue8m0) quantize-dequantize per row (bf16 values in, bf16 out)."""
    x = np.asarray(x, dtype=np.float64)
    blocks = x.reshape(*x.shape[:-1], -1, 32)
    amax = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 1e-4)
    s = e8m0_ceil_pow2(np.float32(amax) * np.float32(1.0 / 448.0))
    q = e4m3_round(np.clip(blocks / s, -448, 448)) * s
    return bf16(q.reshape(x.shape))


def quant_fp4_e8m0(x):
    """fp4_act_quant(block 32, e8m0) quantize-dequantize (the indexer's q and k)."""
    x = np.asarray(x, dtype=np.float64)
    blocks = x.reshape(*x.shape[:-1], -1, 32)
    amax = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 6.0 * 2.0 ** -126)
    s = e8m0_ceil_pow2(np.float32(amax) * np.float32(1.0 / 6.0))
    q = e2m1_round(np.clip(blocks / s, -6, 6)) * s
    return q.reshape(x.shape)


def quant_fp4_e4m3(x):
    """fp4_act_quant(block 16, e4m3 scales) quantize-dequantize (the compressed main KV)."""
    x = np.asarray(x, dtype=np.float64)
    blocks = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 6.0 * 2.0 ** -9)
    s = e4m3_round(amax / 6.0)
    q = e2m1_round(np.clip(blocks / s, -6, 6)) * s
    return bf16(q.reshape(x.shape))


def topk_composite(logits, k):
    """The engine's selection: the k highest fp32 logits, ties to the lower index, ascending."""
    lg = np.asarray(logits, dtype=np.float32)
    order = sorted(range(len(lg)), key=lambda i: (-float(lg[i]), i))[:k]
    return sorted(order)


def sinkhorn_comb(vals, hc_eps, iters):
    """comb = softmax(vals) + eps, one column pass then (iters - 1) row+column passes (double)."""
    n = 4
    c = np.empty((n, n))
    for row in range(n):
        v = vals[row]
        m = v.max()
        ex = np.exp(v - m)
        c[row] = ex / ex.sum()
    c += hc_eps
    def col_pass():
        for col in range(n):
            s = c[:, col].sum()
            c[:, col] = c[:, col] / (s + hc_eps)
    def row_pass():
        for row in range(n):
            s = c[row].sum()
            c[row] = c[row] / (s + hc_eps)
    col_pass()
    for _ in range(iters - 1):
        row_pass()
        col_pass()
    return c


def hc_mixes(x, fn, base, scale, hc_eps, norm_eps, iters):
    """The single-pass mHC coefficients of every token: pre [T, 4], post [T, 4], comb [T, 4, 4] (fp32 exports)."""
    T = x.shape[0]
    flat = x.reshape(T, -1)
    inv_rms = 1.0 / np.sqrt(np.mean(flat * flat, axis=1, keepdims=True) + norm_eps)
    logits = (flat * inv_rms) @ fn.T  # [T, 24]
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    pre = sig(logits[:, 0:4] * scale[0] + base[0:4]) + hc_eps
    post = 2.0 * sig(logits[:, 4:8] * scale[1] + base[4:8])
    comb = np.empty((T, 4, 4))
    for t in range(T):
        vals = logits[t, 8:24].reshape(4, 4) * scale[2] + base[8:24].reshape(4, 4)
        comb[t] = sinkhorn_comb(vals, hc_eps, iters)
    return f32(pre), f32(post), f32(comb)


def hc_pre(x, pre):
    return bf16(np.einsum("tj,tjd->td", pre, x))


def hc_post(y, x, post, comb):
    """x'[i] = bf16(post[i] * y + sum_j comb[j, i] * x[j]) — one rounding."""
    mixed = np.einsum("tji,tjd->tid", comb, x)
    return bf16(post[:, :, None] * y[:, None, :] + mixed)


# ---------------------------------------------------------------------------
# the layers
# ---------------------------------------------------------------------------
def wo_a_group(w, g, o_lora):
    """Group g's wo_a rows: the release's row block, or the pack's per-group matrix."""
    return w.wo_a[g] if isinstance(w.wo_a, list) else w.wo_a[g * o_lora:(g + 1) * o_lora]


def layer_prefix(cfg, layer):
    L = cfg["num_hidden_layers"]
    return ("mtp.%d." % (layer - L)) if layer >= L else ("layers.%d." % layer)


class Weights:
    def __init__(self, cfg, entries, layer):
        L = cfg["num_hidden_layers"]
        self.draft = layer >= L
        p = layer_prefix(cfg, layer)
        self.attn_norm = load_np(entries, p + "attn_norm.weight")
        self.ffn_norm = load_np(entries, p + "ffn_norm.weight")
        self.hc = {}
        for site in ("attn", "ffn"):
            self.hc[site] = (bf16(load_np(entries, p + "hc_%s_fn" % site)), load_np(entries, p + "hc_%s_base" % site),
                             load_np(entries, p + "hc_%s_scale" % site))
        a = p + "attn."
        self.q_norm = load_np(entries, a + "q_norm.weight")
        self.kv_norm = load_np(entries, a + "kv_norm.weight")
        self.sink = load_np(entries, a + "attn_sink")
        self.wq_a = load_w(entries, a + "wq_a")
        self.wq_b = load_w(entries, a + "wq_b")
        self.wkv = load_w(entries, a + "wkv")
        # wo_a is block-diagonal over the output groups; the EXL3 pack ships
        # one matrix per group (wo_a.slice.g, [o_lora, hpg * 512]), stacked
        # here into the release's [o_groups * o_lora, hpg * 512] rows.
        if a + "wo_a.slice.0.trellis" in entries:
            self.wo_a = [load_exl3(entries, a + "wo_a.slice.%d" % g) for g in range(cfg["o_groups"])]
        else:
            self.wo_a = load_w(entries, a + "wo_a")
        self.wo_b = load_w(entries, a + "wo_b")
        self.ratio = cfg["compress_ratios"][layer]
        self.kv_source = layer in cfg["kv_source_layer_ids"]
        self.index_source = layer in cfg["index_source_layer_ids"]
        if self.index_source:
            self.idx_wq_b = load_w(entries, a + "indexer.wq_b")
            self.idx_wp = load_np(entries, a + "indexer.weights_proj.weight")
        if self.kv_source:
            self.idx_wk = (load_exl3(entries, a + "indexer.wk") if a + "indexer.wk.trellis" in entries
                           else load_np(entries, a + "indexer.wk.weight"))
            self.idx_k_norm = load_np(entries, a + "indexer.k_norm.weight")
            self.comp_wkv = (load_exl3(entries, a + "compressor.wkv") if a + "compressor.wkv.trellis" in entries
                             else load_np(entries, a + "compressor.wkv.weight"))
            self.comp_norm = load_np(entries, a + "compressor.norm.weight")
            if self.ratio == 2:
                self.comp_wgate = (load_exl3(entries, a + "compressor.wgate") if a + "compressor.wgate.trellis" in entries
                                   else load_np(entries, a + "compressor.wgate.weight"))
        m = p + "ffn."
        self.router = load_np(entries, m + "gate.weight")
        self.router_bias = load_np(entries, m + "gate.bias")
        E = cfg["dspark_n_routed_experts"] if self.draft else cfg["n_routed_experts"]
        self.top_k = cfg["dspark_num_experts_per_tok"] if self.draft else cfg["num_experts_per_tok"]
        self.experts = [(load_expert(entries, m + "experts.%d.w1" % e), load_expert(entries, m + "experts.%d.w3" % e),
                         load_expert(entries, m + "experts.%d.w2" % e)) for e in range(E)]
        self.shared = (load_w(entries, m + "shared_experts.w1"), load_w(entries, m + "shared_experts.w3"),
                       load_w(entries, m + "shared_experts.w2"))
        self.engram = (not self.draft) and layer in cfg["engram_layer_ids"]
        if self.engram:
            g = p + "engram."
            self.eng_wkv = load_w(entries, g + "wkv")
            self.eng_q = load_np(entries, g + "q_weight")
            self.eng_k = load_np(entries, g + "k_weight")
            self.eng_table = g + "embed"
        if self.draft:
            stage = layer - L
            if stage == 0:
                self.main_proj = load_w(entries, p + "main_proj")
                self.main_norm = load_np(entries, p + "main_norm.weight")
            if stage == cfg["num_nextn_predict_layers"] - 1:
                self.draft_norm = load_np(entries, p + "norm.weight")
                self.markov_embed = load_np(entries, p + "markov_head.embed.weight")
                self.markov_head = load_np(entries, p + "markov_head.head.weight")
                self.confidence = load_np(entries, p + "confidence_head.proj.weight").reshape(-1)


class State:
    """The compressed caches per kv source ordinal and the shared selection / candidates."""
    def __init__(self):
        self.main = {}   # ord -> [entries, 512] (bf16 values)
        self.keys = {}   # ord -> [entries, 128] (fp4 values)
        self.sel = None  # [T] lists
        self.cand = None # [T] lists of block ids
        self.block_margins = None # [T] the candidate source's kept-vs-dropped block gap


def engram_rows(cfg, sc, tokens, layer_index):
    """The 24 table rows per token of one Engram layer: ids [T, 3 * heads]."""
    G, Hh = sc["max_ngram_size"] - 1, sc["n_heads"]
    tm = sc["token_map"]
    mult = sc["multipliers"][layer_index]
    ids = np.zeros((len(tokens), G * Hh), dtype=np.int64)
    for t, tok in enumerate(tokens):
        y = [tm[tokens[t - k]] if t - k >= 0 else sc["pad_class"] for k in range(sc["max_ngram_size"])]
        rolling = (y[0] * mult[0]) & MASK64
        for n in range(1, sc["max_ngram_size"]):
            rolling ^= (y[n] * mult[n]) & MASK64
            mixed = rolling - (1 << 64) if rolling >= (1 << 63) else rolling
            for h in range(Hh):
                prime = sc["primes"][layer_index][n - 1][h]
                off = sc["offsets"][layer_index][n - 1][h]
                ids[t, (n - 1) * Hh + h] = (mixed % prime) + off
    return ids


def engram_apply(cfg, w, entries, sc, tokens, x, layer_index):
    H, hc = cfg["hidden_size"], cfg["hc_mult"]
    ids = engram_rows(cfg, sc, tokens, layer_index)
    T = len(tokens)
    e = np.concatenate([load_table_rows(entries, w.eng_table, ids[:, j]) for j in range(ids.shape[1])], axis=1)  # [T, 24*256]
    kv = linear_bf16(e, w.eng_wkv)  # [T, 5H]
    key = kv[:, : hc * H].reshape(T, hc, H)
    value = kv[:, hc * H:]
    out = np.empty_like(x)
    eps = cfg["rms_norm_eps"]
    for i in range(hc):
        xi = x[:, i, :]
        ki = key[:, i, :]
        rstd = (1.0 / np.sqrt(np.mean(xi * xi, axis=1) + eps)) * (1.0 / np.sqrt(np.mean(ki * ki, axis=1) + eps))
        dot = np.sum(xi * (w.eng_q[i] * w.eng_k[i]) * ki, axis=1) * rstd * (H ** -0.5)
        gate = 1.0 / (1.0 + np.exp(-np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), 1e-6))))
        out[:, i, :] = bf16(xi + gate[:, None] * value)
    return out


def csa2_forward(cfg, w, u, layer, state, freq, T, seg=0, pos0=None):
    """The attention over the prompt (start_pos 0): the window (128), the compressed entries, the selection.
    The bounded prefill (plan §1.8): `seg` > 0 at the split layer — the kv source publishes its entries from
    EVERY row of u, then the attention runs over the rows u[seg:]; `pos0` is the position of u's first row
    after that (the segment's start: the decoder layers after the split take the segment rows at pos0..).
    A segment row sees the segment rows within the window and selects over every published entry."""
    H = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    window = cfg["sliding_window"]
    eps = cfg["rms_norm_eps"]
    if pos0 is None:
        pos0 = seg
    if w.ratio > 0:
        ratio = w.ratio
        ord_ = cfg["kv_source_layer_ids"].index(cfg["_kv_source_of"][layer])
        if w.kv_source:
            if ratio == 1:
                lat = rmsnorm1(linear_bf16(u, w.comp_wkv), w.comp_norm, eps)
            else:
                ckv = f32(u) @ w.comp_wkv.T
                csc = f32(u) @ w.comp_wgate.T
                n = T // 2
                s0, s1 = csc[0:2 * n:2], csc[1:2 * n:2]
                m = np.maximum(s0, s1)
                e0, e1 = np.exp(s0 - m), np.exp(s1 - m)
                pooled = bf16(ckv[0:2 * n:2] * (e0 / (e0 + e1)) + ckv[1:2 * n:2] * (e1 / (e0 + e1)))
                lat = rmsnorm1(pooled, w.comp_norm, eps)
            n_ent = lat.shape[0]
            ent_pos = np.arange(n_ent) * ratio
            ik = rmsnorm1(linear_bf16(lat, w.idx_wk), w.idx_k_norm, eps)
            ik = quant_fp4_e8m0(rope(ik, ent_pos, freq))
            main = quant_fp4_e4m3(rope(lat, ent_pos, freq))
            state.keys[ord_] = ik
            state.main[ord_] = main
    if seg > 0:
        u = u[seg:]
    T = u.shape[0]
    T_full = pos0 + T
    pos = pos0 + np.arange(T)
    qr = rmsnorm1(linear_bf16(u, w.wq_a), w.q_norm, eps)
    q = linear_bf16(qr, w.wq_b).reshape(T, heads, 512)
    q = rope(q, pos, freq)
    kv = rmsnorm1(linear_bf16(u, w.wkv), w.kv_norm, eps)
    kv = rope(kv, pos, freq)
    win = quant_fp8_block(kv)  # the window rows (the segment's)
    sel = [[] for _ in range(T)]
    margins = np.zeros(T)
    all_logits = np.full((T, T_full), np.nan)
    cand_rows = [[] for _ in range(T)]
    block_margins = np.full(T, 1e9)
    if w.ratio > 0:
        keys, main = state.keys[ord_], state.main[ord_]
        if w.index_source:
            iq = linear_bf16(qr, w.idx_wq_b).reshape(T, 32, 128)
            iq = quant_fp4_e8m0(rope(iq, pos, freq))
            wts = bf16(linear_bf16(u, w.idx_wp) * 0.015625)  # [T, 32]
            # The candidate pool and its margins live at absolute positions
            # (a user may run over a segment its source did not).
            cand = [[] for _ in range(T_full)]
            bm_full = np.full(T_full, 1e9)
            for t in range(T):
                visible = (int(pos[t]) + 1) // ratio  # the entries at or before this row's position
                if visible == 0:
                    continue
                dots = np.einsum("hd,jd->hj", iq[t], keys[:visible])
                logits = f32(np.sum(wts[t][:, None] * np.maximum(dots, 0.0), axis=0))
                all_logits[t, :visible] = logits
                if layer == cfg["candidate_source_layer_id"]:
                    bs = cfg["candidate_block_size"]
                    nb = visible // bs
                    scores = np.array([np.max(logits[b * bs:(b + 1) * bs]) for b in range(nb)])
                    if visible % bs == 0 and nb > 0:
                        scores[nb - 1] = np.inf
                    order_b = sorted(range(nb), key=lambda b: (-scores[b], b))
                    kb = cfg["candidate_topk_blocks"]
                    top = order_b[:kb]
                    cand[pos0 + t] = sorted(top)
                    state.cand = cand
                    state.block_margins = bm_full
                    if nb > kb:
                        lo_b, hi_b = float(np.min(logits)), float(np.max(logits))
                        kept_min = min(scores[b] for b in top if np.isfinite(scores[b])) if any(np.isfinite(scores[b]) for b in top) else np.inf
                        bm_full[pos0 + t] = (kept_min - scores[order_b[kb]]) / max(hi_b - lo_b, abs(hi_b), 1e-30)
                pool = list(range(visible))
                if layer == cfg["candidate_source_layer_id"] or cfg["_uses_candidates"][layer]:
                    bs = cfg["candidate_block_size"]
                    blocks = state.cand[pos0 + t]
                    cand_rows[t] = list(blocks)
                    # A user inherits the source's block-level margin: its pool
                    # difference (if any) was decided there.
                    block_margins[t] = state.block_margins[pos0 + t]
                    pool = [b * bs + k for b in blocks for k in range(bs)] + list(range((visible // bs) * bs, visible))
                pool_logits = np.array([logits[e] for e in pool], dtype=np.float32)
                order = sorted(range(len(pool)), key=lambda i: (-float(pool_logits[i]), pool[i]))
                k = min(cfg["index_topk"], len(pool))
                chosen = sorted(pool[i] for i in order[:k])
                sel[t] = chosen
                if len(pool) > k:
                    lo, hi = float(pool_logits.min()), float(pool_logits.max())
                    margins[t] = (pool_logits[order[k - 1]] - pool_logits[order[k]]) / max(hi - lo, abs(hi), 1e-30)
                else:
                    margins[t] = 1e9
            state.sel = sel
        else:
            sel = state.sel
    o = np.empty((T, heads, 512))
    for t in range(T):
        rows = [win[j] for j in range(max(0, t - window + 1), t + 1)]
        if w.ratio > 0:
            rows += [state.main[ord_][e] for e in sel[t]]
        R = np.stack(rows)
        sc = (q[t] @ R.T) * (512 ** -0.5)  # [heads, n]
        m = sc.max(axis=1, keepdims=True)
        pr = np.exp(sc - m)
        den = pr.sum(axis=1) + np.exp(w.sink - m[:, 0])
        # sparse_attn rounds the probabilities to bf16 before the PV product
        # (the denominator keeps the unrounded sum).
        o[t] = bf16((bf16(pr) @ R) / den[:, None])
    o = rope(o, pos, freq, inverse=True)
    hpg = heads // cfg["o_groups"]
    oa = []
    for g in range(cfg["o_groups"]):
        og = o[:, g * hpg:(g + 1) * hpg, :].reshape(T, hpg * 512)
        wg = wo_a_group(w, g, cfg["o_lora_rank"])
        oa.append(linear_bf16(og, wg))
    oa = np.concatenate(oa, axis=1)
    y = linear_bf16(oa, w.wo_b)
    return y, sel, margins, all_logits, cand_rows, block_margins


def moe_forward(cfg, w, u):
    """The engine's chain: bf16 gate/up, the asymmetric clamps, two-rounding swiglu, fp32 downs summed
    with the routing weights and the shared expert, one bf16 rounding."""
    T = u.shape[0]
    K = w.top_k
    lim = cfg["swiglu_limit"]
    scores = f32(u) @ w.router.T
    scores = np.sqrt(np.log1p(np.exp(np.minimum(scores, 80.0))) + np.where(scores > 80.0, scores - np.log1p(np.exp(np.minimum(scores, 80.0))), 0.0))
    biased = scores + w.router_bias
    ids = np.zeros((T, K), dtype=np.int64)
    margins = np.zeros(T)
    out = np.empty((T, cfg["hidden_size"]))
    for t in range(T):
        order = sorted(range(len(biased[t])), key=lambda e: (-biased[t][e], e))
        chosen = sorted(order[:K])
        ids[t] = chosen
        margins[t] = biased[t][order[K - 1]] - biased[t][order[K]] if len(order) > K else 1e9
        ws = scores[t][chosen]
        ws = ws / (ws.sum() + 1e-20) * cfg["routed_scaling_factor"]
        acc = np.zeros(cfg["hidden_size"])
        for e, we in zip(chosen, ws):
            w1, w3, w2 = w.experts[e]
            g = bf16(u[t] @ w1.T)
            up = np.clip(bf16(u[t] @ w3.T), -lim, lim)
            g = np.minimum(g, lim)
            act = bf16(bf16(g / (1.0 + np.exp(-g))) * up)
            acc += we * (act @ w2.T)
        s1, s3, s2 = w.shared
        g = np.minimum(bf16(u[t] @ s1.T), lim)
        up = np.clip(bf16(u[t] @ s3.T), -lim, lim)
        act = bf16(bf16(g / (1.0 + np.exp(-g))) * up)
        acc += act @ s2.T
        out[t] = bf16(acc)
    return out, ids, margins


def load_engine_states(path):
    with open(path, "rb") as f:
        L, T, W = struct.unpack("<iii", f.read(12))
        raw = np.frombuffer(f.read(L * T * W * 2), dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).astype(np.float64).reshape(L, T, W)


def reference_forward(cfg, entries, sc, tokens, teacher=None, bounded=False):
    """The forty-layer walk over the prompt. `bounded` (plan §1.8, the engine's production prefill): the
    layers from the last kv source on (the decoder) run over the last `sliding_window` rows only — the
    segment — after that source published its entries from every row; the decoder layers' states, routes,
    selections and the head's rows are the segment's (the caller pads them to T rows). The teacher's states
    are T rows per layer (a decoder layer's padded before the segment)."""
    T = len(tokens)
    H, hc = cfg["hidden_size"], cfg["hc_mult"]
    L = cfg["num_hidden_layers"]
    dec0 = cfg["kv_source_layer_ids"][-1] if cfg["kv_source_layer_ids"] else L
    seg0 = T - min(cfg["sliding_window"], T) if bounded else 0
    if bounded:
        for e in cfg.get("engram_layer_ids", []):
            assert e < dec0, "the bounded prefill needs the Engram layers inside the encoder"
        for t_ in cfg.get("dspark_target_layer_ids", []):
            assert t_ >= dec0, "the bounded prefill needs the DSpark targets inside the decoder"
    embed = load_np(entries, "embed.weight")
    x = np.repeat(embed[np.array(tokens)][:, None, :], hc, axis=1)  # [T, 4, H]
    pre_mix = np.zeros((T, hc))
    pre_mix[:, 0] = 1.0
    freq_win = inv_freq(cfg["qk_rope_head_dim"], cfg["rope_theta"], 0, cfg["rope_scaling"]["factor"],
                        cfg["rope_scaling"].get("beta_fast", 32.0), cfg["rope_scaling"].get("beta_slow", 1.0))
    freq_comp = inv_freq(cfg["qk_rope_head_dim"], cfg["compress_rope_theta"], cfg["rope_scaling"]["original_max_position_embeddings"],
                         cfg["rope_scaling"]["factor"], cfg["rope_scaling"].get("beta_fast", 32.0),
                         cfg["rope_scaling"].get("beta_slow", 1.0))
    state = State()
    layer_states, routes, route_margins, selections, sel_margins, index_logits = [], [], [], [], [], []
    candidates, block_margins = [], []
    main_hiddens = []
    eps_hc = cfg["hc_eps"]
    iters = cfg["hc_sinkhorn_iters"]
    for layer in range(L):
        w = Weights(cfg, entries, layer)
        # The rows this layer's INPUT spans: every row through the split
        # layer (it publishes from all of them), the segment after.
        lo = seg0 if (bounded and layer > dec0) else 0
        if teacher is not None and layer > 0:
            # Teacher-forced: this layer runs on the ENGINE's output of the
            # previous layer (the per-layer isolation of the cross-check).
            x = teacher[layer - 1].reshape(T, hc, H)[lo:].copy()
        split = bounded and layer == dec0 and seg0 > 0
        if w.engram:
            x = engram_apply(cfg, w, entries, sc, tokens, x, cfg["engram_layer_ids"].index(layer))
        if layer in cfg.get("dspark_target_layer_ids", []):
            # The DSpark target hidden: the stream mean of the attention
            # input (fp32 sum in stream order, one bf16 rounding).
            main_hiddens.append(bf16(np.sum(f32(x), axis=1) * (1.0 / hc)))
        fn, base, scale = w.hc["attn"]
        pre_a, post_a, comb_a = hc_mixes(x, fn, base, scale, eps_hc, cfg["rms_norm_eps"], iters)
        u = rmsnorm1(hc_pre(x, pre_mix), w.attn_norm, cfg["rms_norm_eps"])
        y, sel, margins, lg, cand_rows, bmarg = csa2_forward(cfg, w, u, layer, state, freq_comp if w.ratio > 0 else freq_win, T,
                                                            seg0 if split else 0, seg0 if (bounded and layer >= dec0) else 0)
        if split:
            # The segment: the encoder output's last rows and this site's
            # coefficients of them (the source published from every row).
            x, pre_a, post_a, comb_a = x[seg0:], pre_a[seg0:], post_a[seg0:], comb_a[seg0:]
        x = hc_post(y, x, post_a, comb_a)
        if w.index_source:
            selections.append(sel)
            sel_margins.append(margins)
            index_logits.append(lg)
            candidates.append(cand_rows)
            block_margins.append(bmarg)
        fn, base, scale = w.hc["ffn"]
        pre_f, post_f, comb_f = hc_mixes(x, fn, base, scale, eps_hc, cfg["rms_norm_eps"], iters)
        u = rmsnorm1(hc_pre(x, pre_a), w.ffn_norm, cfg["rms_norm_eps"])
        y, ids, rm = moe_forward(cfg, w, u)
        x = hc_post(y, x, post_f, comb_f)
        pre_mix = pre_f
        routes.append(ids)
        route_margins.append(rm)
        layer_states.append(x.copy())
        print("layer %d done" % layer, file=sys.stderr)
    h = rmsnorm1(hc_pre(x, pre_mix), load_np(entries, "norm.weight"), cfg["rms_norm_eps"])
    logits = f32(h @ head_weight(entries).T)
    main_hidden = np.concatenate(main_hiddens, axis=1) if main_hiddens else None
    return (layer_states, h, logits, routes, route_margins, selections, sel_margins, index_logits, candidates,
            block_margins, main_hidden, seg0)


def dspark_forward(cfg, entries, main_hidden, logits, T, seg0=0):
    """The reference `forward_spec` after the prompt (start_pos T - 1): the draft rings seeded with every
    prompt row's main_x latent, the block [next, noise x (B - 1)] at positions T .. T + B - 1 through the draft
    stages (the bidirectional attention over the ring's last `window` rows and the block), the shared head with
    the draft's norm, the greedy Markov chain and the confidence logits."""
    H, hc = cfg["hidden_size"], cfg["hc_mult"]
    L = cfg["num_hidden_layers"]
    S = cfg["num_nextn_predict_layers"]
    B = cfg["dspark_block_size"]
    heads = cfg["num_attention_heads"]
    window = cfg["sliding_window"]
    eps = cfg["rms_norm_eps"]
    eps_hc = cfg["hc_eps"]
    iters = cfg["hc_sinkhorn_iters"]
    freq = inv_freq(cfg["qk_rope_head_dim"], cfg["rope_theta"], 0, cfg["rope_scaling"]["factor"],
                    cfg["rope_scaling"].get("beta_fast", 32.0), cfg["rope_scaling"].get("beta_slow", 1.0))
    stages = [Weights(cfg, entries, L + s) for s in range(S)]
    w0, wl = stages[0], stages[-1]
    next_id = int(np.argmax(logits[-1]))  # the prompt's last row (the walked rows' last)
    # The prompt rows the walk carried target hidden for (bounded: the
    # segment, at positions seg0..): their main_x seeds the rings.
    main_x = rmsnorm1(linear_bf16(main_hidden, w0.main_proj), w0.main_norm, eps)  # [T - seg0, H]
    ring_pos = seg0 + np.arange(main_x.shape[0])
    # The block.
    ids = [next_id]
    block_tokens = [next_id] + [cfg["dspark_noise_token_id"]] * (B - 1)
    pos = np.arange(T, T + B)
    embed = load_np(entries, "embed.weight")
    x = np.repeat(embed[np.array(block_tokens)][:, None, :], hc, axis=1)  # [B, hc, H]
    pre_mix = np.zeros((B, hc))
    pre_mix[:, 0] = 1.0
    first = max(0, main_x.shape[0] - window)
    sites = []
    for w in stages:
        # This stage's ring: the prompt rows' latents from main_x (the last `window` visible).
        ring = quant_fp8_block(rope(rmsnorm1(linear_bf16(main_x, w.wkv), w.kv_norm, eps), ring_pos, freq))[first:]
        fn, base, scale = w.hc["attn"]
        pre_a, post_a, comb_a = hc_mixes(x, fn, base, scale, eps_hc, eps, iters)
        u = rmsnorm1(hc_pre(x, pre_mix), w.attn_norm, eps)
        qr = rmsnorm1(linear_bf16(u, w.wq_a), w.q_norm, eps)
        q = rope(linear_bf16(qr, w.wq_b).reshape(B, heads, 512), pos, freq)
        kvb = quant_fp8_block(rope(rmsnorm1(linear_bf16(u, w.wkv), w.kv_norm, eps), pos, freq))
        R = np.concatenate([ring, kvb], axis=0)  # every block row attends the ring and the whole block
        o = np.empty((B, heads, 512))
        for t in range(B):
            sc = (q[t] @ R.T) * (512 ** -0.5)
            m = sc.max(axis=1, keepdims=True)
            pr = np.exp(sc - m)
            den = pr.sum(axis=1) + np.exp(w.sink - m[:, 0])
            o[t] = bf16((bf16(pr) @ R) / den[:, None])
        o = rope(o, pos, freq, inverse=True)
        hpg = heads // cfg["o_groups"]
        oa = []
        for g in range(cfg["o_groups"]):
            og = o[:, g * hpg:(g + 1) * hpg, :].reshape(B, hpg * 512)
            wg = wo_a_group(w, g, cfg["o_lora_rank"])
            oa.append(linear_bf16(og, wg))
        y = linear_bf16(np.concatenate(oa, axis=1), w.wo_b)
        site = {"x_attn": u.copy(), "attn_out": y.copy()}
        x = hc_post(y, x, post_a, comb_a)
        fn, base, scale = w.hc["ffn"]
        pre_f, post_f, comb_f = hc_mixes(x, fn, base, scale, eps_hc, eps, iters)
        u = rmsnorm1(hc_pre(x, pre_a), w.ffn_norm, eps)
        y, _, _ = moe_forward(cfg, w, u)
        site["x_ffn"] = u.copy()
        site["ffn_out"] = y.copy()
        sites.append(site)
        x = hc_post(y, x, post_f, comb_f)
        pre_mix = pre_f
    xh = hc_pre(x, pre_mix)  # [B, H] bf16: the confidence head's hidden
    h = rmsnorm1(xh, wl.draft_norm, eps)
    blogits = f32(h @ head_weight(entries).T)  # [B, V]
    margins = np.zeros(B)
    conf = np.zeros(B)
    for i in range(B):
        e = wl.markov_embed[ids[i]]
        blogits[i] = blogits[i] + f32(e) @ f32(wl.markov_head).T
        order = sorted(range(blogits.shape[1]), key=lambda v: (-float(blogits[i][v]), v))
        ids.append(order[0])
        margins[i] = float(blogits[i][order[0]] - blogits[i][order[1]])
        conf[i] = float(np.dot(f32(wl.confidence[:H]), f32(xh[i])) + np.dot(f32(wl.confidence[H:]), f32(e)))
    return blogits, ids, margins, conf, sites


# ---------------------------------------------------------------------------
# the dump
# ---------------------------------------------------------------------------
def bf16_bytes(a):
    v = np.ascontiguousarray(np.asarray(a, dtype=np.float32)).view(np.uint32)
    v = ((v + 0x7FFF + ((v >> 16) & 1)) >> 16).astype(np.uint16)
    return v.tobytes()


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-dsv41-reference-dump", "version": VERSION, "backend": "pure", "config": cfg_summary,
              "tensors": {}}
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


def prepare_config(cfg):
    L = cfg["num_hidden_layers"]
    kv_of, uses = {}, {}
    for l in range(L):
        best = -1
        for s in cfg["kv_source_layer_ids"]:
            if s <= l:
                best = s
        kv_of[l] = best if cfg["compress_ratios"][l] > 0 else -1
        cs = cfg.get("candidate_source_layer_id", -1)
        uses[l] = 0 <= cs < l and l in cfg["index_source_layer_ids"]
    cfg["_kv_source_of"] = kv_of
    cfg["_uses_candidates"] = uses
    cfg.setdefault("candidate_source_layer_id", -1)
    return cfg


def gen_pure(args):
    cfg, _ = text_config(args.checkpoint_dir)
    cfg = prepare_config(cfg)
    entries = read_safetensors_index(args.checkpoint_dir)
    sc = load_sidecar(args.checkpoint_dir)
    rng = np.random.default_rng(args.seed)
    tokens = [int(v) for v in rng.integers(0, cfg["vocab_size"], size=args.tokens)]
    teacher = load_engine_states(args.teacher) if args.teacher else None
    bounded = args.prefill == "bounded"
    (layer_states, h, logits, routes, route_margins, selections, sel_margins,
     index_logits, candidates, block_margins, main_hidden, seg0) = reference_forward(cfg, entries, sc, tokens, teacher, bounded)
    dspark = None
    if main_hidden is not None and cfg.get("num_nextn_predict_layers", 0) > 0 and cfg.get("dspark_block_size", 0) > 0:
        dspark = dspark_forward(cfg, entries, main_hidden, logits, args.tokens, seg0)
    T, H, hc = args.tokens, cfg["hidden_size"], cfg["hc_mult"]
    L, K, topk = cfg["num_hidden_layers"], cfg["num_experts_per_tok"], 8
    S = cfg["index_topk"]

    # The bounded walk's decoder rows are the segment's: every per-row
    # tensor pads to T rows (NaN / -1 before the segment; the rows an
    # engine in bounded mode never computes).
    def pad_rows(a, fill):
        n = T - a.shape[0]
        if n <= 0:
            return a
        return np.concatenate([np.full((n,) + a.shape[1:], fill, dtype=a.dtype), a], axis=0)

    def pad_list(rows):
        return [[] for _ in range(T - len(rows))] + list(rows)

    layer_states = [pad_rows(s_, np.nan) for s_ in layer_states]
    h = pad_rows(h, np.nan)
    logits = pad_rows(logits, np.nan)
    routes = [pad_rows(r_, -1) for r_ in routes]
    route_margins = [pad_rows(np.asarray(m_, dtype=np.float64), np.nan) for m_ in route_margins]
    selections = [pad_list(s_) for s_ in selections]
    sel_margins = [pad_rows(np.asarray(m_, dtype=np.float64), np.nan) for m_ in sel_margins]
    index_logits = [pad_rows(lg, np.nan) for lg in index_logits]
    candidates = [pad_list(c_) for c_ in candidates]
    block_margins = [pad_rows(np.asarray(m_, dtype=np.float64), np.nan) for m_ in block_margins]
    top_ids, top_vals = [], []
    for row in logits:
        if not np.all(np.isfinite(row)):
            top_ids.extend([-1] * topk)
            top_vals.extend([float("nan")] * topk)
            continue
        order = sorted(range(len(row)), key=lambda i: (-float(row[i]), i))[:topk]
        top_ids.extend(order)
        top_vals.extend(float(row[i]) for i in order)
    Li = len(selections)
    flat_sel = []
    for layer in selections:
        for row in layer:
            flat_sel.extend(list(row) + [-1] * (S - len(row)))
    flat_smarg = [float(m) for layer in sel_margins for m in layer]
    CB = cfg["candidate_topk_blocks"]
    flat_cand = []
    for layer in candidates:
        for row in layer:
            flat_cand.extend(list(row) + [-1] * (CB - len(row)))
    flat_bmarg = [float(m) for layer in block_margins for m in layer]
    flat_routes = [int(e) for layer in routes for row in layer for e in row]
    flat_rmarg = [float(m) for layer in route_margins for m in layer]
    tensors = {
        "tokens": ("I64", [T], struct.pack("<%dq" % T, *tokens)),
        "final_hidden": ("BF16", [T, H], bf16_bytes(h)),
        "topk_ids": ("I32", [T, topk], struct.pack("<%di" % (T * topk), *top_ids)),
        "topk_logits": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *top_vals)),
        "layer_states": ("BF16", [L, T, hc * H], bf16_bytes(np.stack(layer_states).reshape(L, T, hc * H))),
        "route_ids": ("I32", [L, T, K], struct.pack("<%di" % len(flat_routes), *flat_routes)),
        "route_margins": ("F32", [L, T], struct.pack("<%df" % len(flat_rmarg), *flat_rmarg)),
        "csa2_selections": ("I32", [Li, T, S], struct.pack("<%di" % len(flat_sel), *flat_sel)),
        "csa2_margins": ("F32", [Li, T], struct.pack("<%df" % len(flat_smarg), *flat_smarg)),
        "csa2_logits": ("F32", [Li, T, T], np.stack(index_logits).astype(np.float32).tobytes() if Li else b""),
        "csa2_candidates": ("I32", [Li, T, CB], struct.pack("<%di" % len(flat_cand), *flat_cand)),
        "csa2_block_margins": ("F32", [Li, T], struct.pack("<%df" % len(flat_bmarg), *flat_bmarg)),
    }
    cfg_summary = {"hidden": H, "vocab": cfg["vocab_size"], "num_layers": L, "tokens": T, "top_k": topk,
                   "moe_layers": L, "index_layers": Li, "max_selected": S, "streams": hc, "candidate_blocks": CB,
                   "dspark_block": 0, "prefill": args.prefill, "segment_row0": seg0}
    if dspark is not None:
        blogits, ids, margins, conf, sites = dspark
        Bk = cfg["dspark_block_size"]
        Sk = len(sites)
        for key in ("x_attn", "attn_out", "x_ffn", "ffn_out"):
            tensors["dspark_" + key] = ("BF16", [Sk, Bk, H], bf16_bytes(np.stack([st[key] for st in sites])))
        cfg_summary["dspark_block"] = Bk
        tensors["dspark_logits"] = ("F32", [Bk, cfg["vocab_size"]], blogits.astype(np.float32).tobytes())
        tensors["dspark_ids"] = ("I32", [Bk + 1], struct.pack("<%di" % (Bk + 1), *ids))
        tensors["dspark_margins"] = ("F32", [Bk], struct.pack("<%df" % Bk, *[float(m) for m in margins]))
        tensors["dspark_confidence"] = ("F32", [Bk], struct.pack("<%df" % Bk, *[float(c) for c in conf]))
    write_dump(args.out, cfg_summary, tensors)
    print("wrote %s (%d tokens, %d layers, %d index sources, %s prefill%s)" % (
        args.out, T, L, Li, args.prefill, ", segment from row %d" % seg0 if bounded else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-pure")
    g.add_argument("--checkpoint-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tokens", type=int, default=40)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--teacher", help="the engine's layer states (DGPP_DSV41_DUMP_ENGINE_STATES): every layer "
                   "computed from the engine's input to it")
    g.add_argument("--prefill", choices=("exact", "bounded"), default="exact",
                   help="exact: every layer over every row; bounded: the decoder over the last window rows "
                   "(plan 1.8, the engine's production prefill)")
    g.set_defaults(fn=gen_pure)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
