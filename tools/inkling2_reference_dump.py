#!/usr/bin/env python3
"""Inkling 2 (looped) full-model reference dump (docs/inkling2_mini_plan.md).

Computes the Inkling2LoopForConditionalGeneration text forward in numpy from
the SAME checkpoint the engine runs and writes the comparable outputs (the
tokens, the partial state after every layer visit, the attention and MLP
contributions of every visit, the routing decisions, the final hidden state
and per-token top-k logits) as a DGPPINK2 dump read by
tests/cuda/inkling2_forward_test.cpp.

The model has no residual stream. A stack of completed block states and one
partial state feed every sublayer through a learned softmax mix. Layers
[core_start, core_end) run num_passes times from the state saved at
core_start, and passes after the first blend four messages read from the
previous pass into their sublayer inputs. See the plan document, section 1.

Rounding follows the released numerics: every stored row is bf16, the
interior math is wider (float64 here, fp32 in the engine), and a value is
rounded once where it is stored.

  the embedding norm, every sublayer norm      bf16(x * rstd * w), from the unrounded mix
  projections (q, r, gate, sink, k, v, o)      bf16(dot)
  q and r                                      bf16(l2_clip(x * scale)), full layers then bf16(. * tau)
  the relative bias table                      bf16(r @ rel_proj)
  K, V convolutions                            bf16(conv), then bf16(l2_clip) / bf16(rms)
  attention                                    wide softmax with the sink, bf16(out * sigmoid(gate))
  the attention and MLP convolutions           bf16(conv)
  the experts                                  bf16 gate and up, bf16(activation * weight), the sum rounded once
  the partial state                            bf16(partial + attention + mlp)

`--activations native` quantizes both expert GEMM inputs as the checkpoint's
precision metadata requires (NVFP4 rows with the four-over-six block scale
choice, MXFP8 blocks with power-of-two scales). The quantizers below follow
the released kernels operation by operation in float32.

Usage:
  inkling2_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE
      [--tokens T] [--seed S] [--activations bf16|native] [--teacher STATES]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_reference_dump import E2M1_TABLE, bf16, e4m3_decode, e4m3_round  # noqa: E402
from glm_reference_dump import _parse_shard_header  # noqa: E402
from kda_reference_dump import make_rng  # noqa: E402

MAGIC = b"DGPPINK2"
VERSION = 1

MIX_EPS = 1e-6           # the source normalization inside every mix and controller
CLIP_EPS = 1e-6          # added to the L2 norm in the q / k / r clips
ROUND_DOWN = True        # each expert's down output is rounded to bf16 before the sum, as the released kernels store it
STREAMS = ("attn", "k", "v", "mlp")   # the controller order of the message queries


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def bf16_to_bytes(values):
    a = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

def read_index(model_dir):
    """Every tensor of the main checkpoint file(s): name -> (path, offset,
    nbytes, dtype, shape). The draft model's mtp.safetensors is not part of
    the text forward."""
    entries = {}
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors") and name != "mtp.safetensors":
            _parse_shard_header(os.path.join(model_dir, name), entries)
    if not entries:
        raise FileNotFoundError("no safetensors file in %s" % model_dir)
    return entries


def load_np(entries, name):
    """BF16 / F32 as float64 (exact), F8_E4M3 / U8 as uint8 codes."""
    path, offset, nbytes, dtype, shape = entries[name]
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(nbytes)
    if dtype == "BF16":
        w = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return w.view(np.float32).astype(np.float64).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).astype(np.float64).reshape(shape)
    if dtype in ("F8_E4M3", "U8"):
        return np.frombuffer(buf, dtype=np.uint8).reshape(shape)
    raise ValueError("unsupported dtype %s for %s" % (dtype, name))


def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        full = json.load(f)
    tc = full["text_config"]
    loop = tc["loop_config"]
    layers = tc["num_hidden_layers"]
    local = set(tc["local_layer_ids"])
    mxfp8 = set()
    quant_path = os.path.join(checkpoint_dir, "hf_quant_config.json")
    with open(quant_path) as f:
        for group in json.load(f)["model"]["parameter_groups"]:
            if group["compute_dtype"] == "mxfp8":
                for name in group["tensors"]:
                    if name.endswith(".mlp.experts.w13_weight"):
                        mxfp8.add(int(name.split(".")[3]))
    cfg = {
        "hidden": tc["hidden_size"], "layers": layers,
        "vocab": tc["unpadded_vocab_size"], "padded_vocab": tc["vocab_size"],
        "heads": tc["num_attention_heads"], "kv_heads": tc["num_key_value_heads"], "head_dim": tc["head_dim"],
        "d_rel": tc["d_rel"], "rel_extent": tc["rel_extent"], "window": tc["sliding_window_size"],
        "local": [1 if i in local else 0 for i in range(layers)],
        "kernel": tc["sconv_kernel_size"], "eps": tc.get("rms_norm_eps", 1e-6),
        "qk_clip": tc["qk_l2_clip_threshold"], "rel_clip": tc["rel_l2_clip_threshold"],
        "n_floor": tc["log_scaling_n_floor"], "alpha": tc["log_scaling_alpha"],
        "block": tc["n_layers_per_attn_res_block"],
        "experts": tc["n_routed_experts"], "top_k": tc["num_experts_per_tok"], "inter": tc["intermediate_size"],
        "gate_cap": tc["mlp_gate_softcap"], "up_cap": tc["mlp_up_softcap"], "route_scale": tc["route_scale"],
        "logit_cap": tc["final_logit_softcapping"],
        "core_start": loop["core_start"], "core_end": loop["core_end"], "passes": loop["num_passes"],
        "mxfp8": [1 if i in mxfp8 else 0 for i in range(layers)],
        "eos": full.get("eos_token_id", tc.get("eos_token_id")),
    }
    for key in ("swa_num_attention_heads", "swa_num_key_value_heads", "swa_head_dim"):
        base = {"swa_num_attention_heads": "heads", "swa_num_key_value_heads": "kv_heads", "swa_head_dim": "head_dim"}[key]
        if tc.get(key, cfg[base]) != cfg[base]:
            raise ValueError("%s differs from the full-attention geometry: not this reference's contract" % key)
    return cfg


def visits_of(cfg):
    """(layer, pass) in execution order: the prelude, every pass of the core,
    the coda."""
    cs, ce = cfg["core_start"], cfg["core_end"]
    out = [(i, 0) for i in range(cs)]
    out += [(i, p) for p in range(cfg["passes"]) for i in range(cs, ce)]
    out += [(i, 0) for i in range(ce, cfg["layers"])]
    return out


class Experts:
    """One projection of every expert, kept in the checkpoint's bytes and
    decoded one expert at a time to its exact float64 [N, K] `inner` values.
    NVFP4: inner = e2m1 * e4m3 block scale, and a finished dot is multiplied
    by the expert's fp32 factor amax / outer_scale_denominator. MXFP8: inner
    = e4m3 * 2^(scale - 127), no factor."""

    def __init__(self, entries, prefix, mxfp8):
        self.mxfp8 = mxfp8
        self.w = load_np(entries, prefix + "_weight")
        self.s = load_np(entries, prefix + "_weight_scale")
        if not mxfp8:
            self.amax = load_np(entries, prefix + "_weight_amax").astype(np.float32)
            self.den = load_np(entries, prefix + "_weight_outer_scale_denominator").astype(np.float32).reshape(-1)[0]

    def matrix(self, e):
        if self.mxfp8:
            return e4m3_decode(self.w[e]) * np.repeat(np.exp2(self.s[e].astype(np.float64) - 127.0), 32, axis=1)
        w = self.w[e]
        codes = np.empty((w.shape[0], w.shape[1] * 2), dtype=np.intp)
        codes[:, 0::2] = w & 0xF
        codes[:, 1::2] = w >> 4
        return E2M1_TABLE[codes] * np.repeat(e4m3_decode(self.s[e]), 16, axis=1)


def layer_weights(cfg, entries, layer):
    p = "model.llm.layers.%d." % layer
    a = p + "attn."
    local = cfg["local"][layer] == 1
    w = {"local": local, "mxfp8": cfg["mxfp8"][layer] == 1}
    for key, name in (("q", "wq_du"), ("k", "wk_dv"), ("v", "wv_dv"), ("r", "wr_du"), ("gate", "gate_attn"),
                      ("sink", "attn_sink_proj"), ("o", "wo_ud")):
        w[key] = load_np(entries, a + name + ".weight")
    w["q_scale"] = load_np(entries, a + "q_l2_clip.scale")
    w["r_scale"] = load_np(entries, a + "rel_l2_clip.scale")
    w["rel_proj"] = load_np(entries, a + "rel_logits_proj.proj")
    if local:
        w["k_conv"] = load_np(entries, a + "k_sconv.weight")[:, 0, :]
        w["v_conv"] = load_np(entries, a + "v_sconv.weight")[:, 0, :]
        norm = load_np(entries, p + "attn_norm.weight")
        w["norms"] = (norm, norm, norm)
    else:
        w["k_static"] = load_np(entries, a + "k_sconv.static_w")
        w["k_dyn"] = load_np(entries, a + "k_sconv.weight_proj")
        w["v_static"] = load_np(entries, a + "v_sconv.static_w")
        w["v_dyn"] = load_np(entries, a + "v_sconv.weight_proj")
        w["norms"] = tuple(load_np(entries, p + "%s_input_norm.weight" % s) for s in "qkv")
    w["mlp_norm"] = load_np(entries, p + "mlp_norm.weight")
    w["attn_conv"] = load_np(entries, p + "attn_sconv.weight")[:, 0, :]
    w["mlp_conv"] = load_np(entries, p + "mlp_sconv.weight")[:, 0, :]
    w["router"] = load_np(entries, p + "mlp.gate.weight")
    w["router_bias"] = load_np(entries, p + "mlp.gate.bias")
    e = p + "mlp.experts."
    w["w13"] = Experts(entries, e + "w13", w["mxfp8"])
    w["w2"] = Experts(entries, e + "w2", w["mxfp8"])
    if cfg["core_start"] <= layer < cfg["core_end"]:
        c = p + "controllers."
        w["ctrl"] = {s: tuple(load_np(entries, c + "%s.%s_weight" % (s, part)) for part in ("down", "gain", "forget"))
                     for s in STREAMS}
    return w


# ---------------------------------------------------------------------------
# row-wise modules
# ---------------------------------------------------------------------------

def rstd(x, eps):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def rmsnorm(x, w, eps):
    """y = bf16(x * rstd * w), one rounding."""
    return bf16(x * rstd(x, eps) * w)


def mix(stack, last, query, hidden):
    """The attention-residual mix of the completed blocks `stack` (a list of
    [T, H] bf16 rows) and `last` with one query: the softmax over
    (q . x_n) / sqrt(H) * rstd_n, the sum of the normalized sources."""
    x = np.stack(list(stack) + [last], axis=1)                 # [T, N, H]
    r = rstd(x, MIX_EPS)                                       # [T, N, 1]
    logits = (x @ query) * hidden ** -0.5 * r[..., 0]          # [T, N]
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / p.sum(axis=1, keepdims=True)
    return np.einsum("tn,tnh->th", p, x * r)


def blend(dest, message, ctrl):
    """(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message with
    h = bf16(bf16([rms(dest), rms(message)]) @ down^T)."""
    down, gain, forget = ctrl
    pair = np.concatenate([bf16(dest * rstd(dest, MIX_EPS)), bf16(message * rstd(message, MIX_EPS))], axis=-1)
    h = bf16(pair @ down.T)
    return (1.0 - np.tanh(f32(h @ forget.T))) * dest + np.tanh(f32(h @ gain.T)) * message


def l2_clip(x, threshold):
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x * np.minimum(threshold / (norm + CLIP_EPS), 1.0)


def static_conv(x, weight):
    """y[t] = (sum_k w[c, k] x[t - (W - 1) + k] + x[t]) / sqrt(W + 1), the
    rows before position 0 zero. x [T, C] bf16, weight [C, W]."""
    T, C = x.shape
    W = weight.shape[1]
    padded = np.concatenate([np.zeros((W - 1, C)), x], axis=0)
    acc = np.zeros((T, C))
    for k in range(W):
        acc += weight[:, k] * padded[k:k + T]
    return bf16((acc + x) / np.sqrt(W + 1.0))


def dynamic_conv(x, static_w, taps, heads):
    """y[t] = (sum_lag (static_w[lag, c] + taps[t, head(c), lag]) x[t - lag]
    + x[t]) / sqrt(2 W + 1). x [T, C], static_w [W, C], taps [T, heads * W]."""
    T, C = x.shape
    W = static_w.shape[0]
    padded = np.concatenate([np.zeros((W - 1, C)), x], axis=0)
    dyn = taps.reshape(T, heads, W)
    acc = np.zeros((T, C))
    for lag in range(W):
        w = static_w[lag][None, :] + np.repeat(dyn[:, :, lag], C // heads, axis=1)
        acc += w * padded[W - 1 - lag:W - 1 - lag + T]
    return bf16((acc + x) / np.sqrt(2.0 * W + 1.0))


def attention(cfg, w, xq, xk, xv, positions, world=1):
    """One layer's attention on normed bf16 inputs for T rows at absolute
    `positions` (0..T-1 here: the reference runs the whole sequence). A
    tensor-parallel world of `world` ranks projects each rank's heads on
    its own and adds the bf16 partial outputs."""
    T = xq.shape[0]
    Hq, Hkv, D, R = cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["d_rel"]
    local = w["local"]
    # The bias extent is the table's own width: the window on a local layer,
    # rel_extent on a full-attention one.
    E = w["rel_proj"].shape[1]
    q = bf16(xq @ w["q"].T).reshape(T, Hq, D)
    r = bf16(xq @ w["r"].T).reshape(T, Hq, R)
    gate = bf16(xq @ w["gate"].T).reshape(T, Hq, D)
    sink = bf16(xq @ w["sink"].T)                                 # [T, Hq]
    k = bf16(xk @ w["k"].T)                                       # [T, Hkv * D]
    v = bf16(xv @ w["v"].T)
    q = bf16(l2_clip(q * w["q_scale"], cfg["qk_clip"]))
    r = bf16(l2_clip(r * w["r_scale"], cfg["rel_clip"]))
    if not local:
        pos = np.asarray(positions, dtype=np.float32)
        tau = (np.float32(1.0) + np.float32(cfg["alpha"]) *
               np.log(np.maximum((pos + np.float32(1.0)) / np.float32(cfg["n_floor"]), np.float32(1.0)),
                      dtype=np.float32)).astype(np.float64)
        q = bf16(q * tau[:, None, None])
        r = bf16(r * tau[:, None, None])
    bias = bf16(r @ w["rel_proj"])                                # [T, Hq, E]
    if local:
        k = static_conv(k, w["k_conv"])
        v = static_conv(v, w["v_conv"])
    else:
        k = dynamic_conv(k, w["k_static"], bf16(xk @ w["k_dyn"].T), Hkv)
        v = dynamic_conv(v, w["v_static"], bf16(xv @ w["v_dyn"].T), Hkv)
    K = bf16(l2_clip(k.reshape(T, Hkv, D), cfg["qk_clip"]))
    vh = v.reshape(T, Hkv, D)
    V = bf16(vh * rstd(vh, cfg["eps"]))
    rep = Hq // Hkv
    out = np.zeros((T, Hq, D))
    window = cfg["window"]
    for t in range(T):
        lo = max(0, t - window + 1) if local else 0
        dist = t - np.arange(lo, t + 1)
        in_extent = dist < E
        for h in range(Hq):
            kv = h // rep
            logit = (K[lo:t + 1, kv, :] @ q[t, h]) * D ** -0.5
            logit = logit + np.where(in_extent, bias[t, h, np.minimum(dist, E - 1)], 0.0)
            m = max(float(logit.max()), float(sink[t, h]))
            e = np.exp(logit - m)
            out[t, h] = (e / (e.sum() + np.exp(sink[t, h] - m))) @ V[lo:t + 1, kv, :]
    gated = bf16(out * sigmoid(gate)).reshape(T, Hq * D)
    cols = Hq * D // world
    return bf16(sum(bf16(gated[:, r * cols:(r + 1) * cols] @ w["o"][:, r * cols:(r + 1) * cols].T) for r in range(world)))


# ---------------------------------------------------------------------------
# the experts
# ---------------------------------------------------------------------------

F32_MAX = np.float32(3.4028234663852886e38)


def e2m1_round(x):
    """Round-to-nearest-even, saturating: the code (sign in bit 3) and the
    value, the released kernels' table."""
    a = np.abs(x)
    code = np.where(a <= 0.25, 0, np.where(a < 0.75, 1, np.where(a <= 1.25, 2, np.where(a < 1.75, 3, np.where(
        a <= 2.5, 4, np.where(a < 3.5, 5, np.where(a <= 5.0, 6, 7))))))).astype(np.int64)
    sign = np.signbit(x)
    return code | (sign.astype(np.int64) << 3), E2M1_TABLE[code | (sign.astype(np.int64) << 3)]


def nvfp4_quantize_row(x):
    """The four-over-six NVFP4 quantization of one bf16 row (float32 math in
    the released kernels' operation order). Returns the exact inner values
    (e2m1 * e4m3 block scale, float64) and the fp32 row amax: the row is
    inner * (amax / 1536)."""
    x = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(x)).astype(np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        encoder = np.minimum(np.float32(1536.0) / amax, F32_MAX).astype(np.float32)
        decoder = (np.float32(1.0) / encoder).astype(np.float32)
        outer = (amax * np.float32(1.0 / 1536.0)).astype(np.float32)
        blocks = x.reshape(-1, 16)
        block_max = np.max(np.abs(blocks), axis=1).astype(np.float32)
        inner = np.empty(blocks.shape, dtype=np.float64)
        candidates = []
        for factor in (np.float32(1.0 / 6.0), np.float32(0.25)):
            raw = np.minimum(block_max * (encoder * factor).astype(np.float32), np.float32(448.0)).astype(np.float32)
            scale = e4m3_round(raw).astype(np.float32)
            inverse = np.minimum(np.float32(1.0) / (scale * decoder).astype(np.float32), F32_MAX).astype(np.float32)
            _, value = e2m1_round((blocks * inverse[:, None]).astype(np.float32))
            recon = ((value.astype(np.float32) * scale[:, None]).astype(np.float32) * outer).astype(np.float32)
            err = np.abs(recon - blocks).astype(np.float32)
            for width in (8, 4, 2, 1):                           # the pairwise error sum
                err = (err.reshape(-1, width, 2)[:, :, 0] + err.reshape(-1, width, 2)[:, :, 1]).astype(np.float32)
            candidates.append((value.astype(np.float64) * scale[:, None].astype(np.float64), err.reshape(-1)))
    (six, err6), (four, err4) = candidates
    choose4 = err4 < err6
    inner = np.where(choose4[:, None], four, six)
    return inner.reshape(-1), amax


def mxfp8_quantize_row(x):
    """MXFP8: per 32 values a power-of-two scale that covers the block
    maximum (rounded up) and e4m3 values. Returns the exact dequantized row."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    amax = np.max(np.abs(x), axis=1).astype(np.float32)
    raw = (amax * np.float32(1.0 / 448.0)).astype(np.float32)
    bits = raw.view(np.uint32).astype(np.int64)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = np.where(raw <= np.float32(2.0 ** -127), 0, np.minimum(exponent, 254))
    scale = np.exp2(np.maximum(exponent, 0).astype(np.float64) - 127.0)
    scale = np.where(exponent == 0, 2.0 ** -127, scale)
    values = np.clip(x.astype(np.float64) / scale[:, None], -448.0, 448.0)
    return (e4m3_round(values) * scale[:, None]).reshape(-1)


def softcap(x, cap):
    return cap * np.tanh(x / cap)


def quantize_rows(rows, mxfp8):
    """The native activation recipe over rows: the exact inner rows and, for
    NVFP4, each row's fp32 amax."""
    if mxfp8:
        return np.stack([mxfp8_quantize_row(r) for r in rows]), None
    pairs = [nvfp4_quantize_row(r) for r in rows]
    return np.stack([inner for inner, _ in pairs]), np.array([amax for _, amax in pairs], dtype=np.float32)


def moe(cfg, w, x, native, world=1):
    """The routed experts on normed bf16 rows x [T, H]. Returns the bf16
    output and the picked expert ids (ascending) per row. Rows are grouped by
    expert, and a row's outputs are summed in ascending expert id.

    A tensor-parallel world of `world` ranks slices every expert's
    intermediate width: a rank holds one slice of the activation, quantizes
    it on its own under the native recipe (its own row maximum), sums its
    slice of the down projection and rounds, and the ranks' bf16 partial sums
    are added. One rank is the released recipe."""
    T, H = x.shape
    E, K = cfg["experts"], cfg["top_k"]
    logits = f32(x @ w["router"].T)
    scores = sigmoid(logits)
    key = f32(scores + w["router_bias"])
    routes = []
    by_expert = {}
    for t in range(T):
        order = sorted(range(E), key=lambda e: (-key[t, e], e))[:K]
        picked = sorted(order)
        routes.append(picked)
        total = sum(scores[t, e] for e in picked) + 1e-20
        for e in picked:
            weight = np.float64(np.float32(scores[t, e] / total * cfg["route_scale"]))
            by_expert.setdefault(e, []).append((t, weight))
    mx = w["mxfp8"]
    a13, x_amax = quantize_rows(x, mx) if native else (x, None)
    partial = [np.zeros((T, H)) for _ in range(world)]
    for e in sorted(by_expert):
        rows = [t for t, _ in by_expert[e]]
        weights = np.array([weight for _, weight in by_expert[e]])
        gu = project(a13[rows], None if x_amax is None else x_amax[rows], w["w13"], e)
        gate, up = gu[:, 0::2], gu[:, 1::2]                        # the checkpoint interleaves (gate, up) rows
        act = bf16(softcap(gate, cfg["gate_cap"]) * sigmoid(gate) * softcap(up, cfg["up_cap"]) * weights[:, None])
        down = w["w2"].matrix(e)
        width = act.shape[1] // world
        for r in range(world):
            cols = slice(r * width, (r + 1) * width)
            a2, act_amax = quantize_rows(act[:, cols], mx) if native else (act[:, cols], None)
            partial[r][rows] += project(a2, act_amax, w["w2"], e, rounded=ROUND_DOWN, matrix=down[:, cols])
    if world == 1:
        return bf16(partial[0]), routes
    return bf16(sum(bf16(p) for p in partial)), routes


def project(act, act_amax, table, e, rounded=True, matrix=None):
    """One expert projection of rows act [n, K]: the exact inner dot times the
    fp32 scale. bf16 activations: scale = amax_e / denominator. Native NVFP4
    (act_amax set): scale = (act_amax * amax_e) * rcp(1536 * denominator), the
    released row scale. MXFP8: none. `matrix`: a column slice of the
    expert's matrix in place of the whole."""
    dot = f32(act @ (table.matrix(e) if matrix is None else matrix).T)
    if not table.mxfp8:
        if act_amax is not None:
            scale = ((act_amax.astype(np.float32) * table.amax[e]).astype(np.float32) *
                     (np.float32(1.0) / (np.float32(1536.0) * table.den))).astype(np.float32)
            dot = f32(dot * scale.astype(np.float64)[:, None])
        else:
            dot = f32(dot * np.float64(np.float32(table.amax[e] / table.den)))
    return bf16(dot) if rounded else dot


# ---------------------------------------------------------------------------
# the forward
# ---------------------------------------------------------------------------

def load_engine_states(path):
    """The engine's partial state after every visit and its expert input in
    every visit (int32 V, T, H, then two blocks of bf16 [V, T, H]). The
    teacher-forced mode starts each reference visit from the engine's own
    state and runs the experts on the engine's own input: the activation
    quantizers are discontinuous, so a one-ulp input difference would move
    an expert output by whole quantization steps."""
    with open(path, "rb") as f:
        V, T, H = struct.unpack("<iii", f.read(12))
        raw = np.frombuffer(f.read(), dtype=np.uint16)
    if raw.size != 2 * V * T * H:
        raise ValueError("%s: expected the states and the expert inputs of %d visits" % (path, V))
    rows = (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(2, V, T, H)
    return [rows[0, v] for v in range(V)], [rows[1, v] for v in range(V)]


def reference_forward(cfg, entries, tokens, native=False, teacher=None, progress=False, world=1):
    """Returns the per-visit partial states, attention contributions, MLP
    contributions, expert inputs and routes, the final hidden state and the
    logits. `teacher` is load_engine_states' pair."""
    teacher, teacher_mlp_in = teacher if teacher is not None else (None, None)
    T = len(tokens)
    H, B = cfg["hidden"], cfg["block"]
    cs, ce = cfg["core_start"], cfg["core_end"]
    positions = np.arange(T)
    embed = load_np(entries, "model.llm.embed.weight")
    partial = rmsnorm(embed[np.asarray(tokens)], load_np(entries, "model.llm.embed_norm.weight"), cfg["eps"])
    attn_q = {i + 1: load_np(entries, "model.llm.attn_res.attn_queries.%d" % i) for i in range(cfg["layers"] - 1)}
    mlp_q = [load_np(entries, "model.llm.attn_res.mlp_queries.%d" % i) for i in range(cfg["layers"])]
    out_q = load_np(entries, "model.llm.attn_res.output_query")
    cross = load_np(entries, "model.llm.middle_cycle_cross_pass_queries").reshape(-1, 2, H)
    cross_kv = load_np(entries, "model.llm.middle_cycle_cross_pass_kv_queries").reshape(-1, 2, H)
    # per core layer: (attn, k, v, mlp)
    msg_q = np.stack([cross[:, 0], cross_kv[:, 0], cross_kv[:, 1], cross[:, 1]], axis=1)

    weights = {}
    stack = []
    prelude = None
    bank = None
    visit_states, visit_attn, visit_mlp, visit_mlp_in, routes = [], [], [], [], []
    for visit, (layer, p) in enumerate(visits_of(cfg)):
        if layer not in weights:
            weights[layer] = layer_weights(cfg, entries, layer)
        w = weights[layer]
        if layer == cs:
            if p == 0:
                prelude = (list(stack), partial.copy())
            else:
                # the previous pass's blocks after the prelude's own, then its partial
                bank = (stack[len(prelude[0]) + 1:], partial)
                stack, partial = list(prelude[0]), prelude[1].copy()
        messages = None
        if p > 0:
            messages = [mix(bank[0], bank[1], msg_q[layer - cs][s], H) for s in range(4)]
        if progress:
            print("visit %d: layer %d pass %d (%s, %s)" % (visit, layer, p, "local" if w["local"] else "full",
                                                         "mxfp8" if w["mxfp8"] else "nvfp4"), file=sys.stderr, flush=True)
        # attention inputs
        if layer == 0:
            dests = [partial, partial, partial]
        elif w["local"]:
            d = mix(stack, partial, attn_q[layer], H)
            dests = [d, d, d]
        else:
            dests = [mix(stack, partial, attn_q[layer][j], H) for j in range(3)]
        if messages is not None:
            dests = [blend(dests[j], messages[j], w["ctrl"][STREAMS[j]]) for j in range(3)]
        xq, xk, xv = (rmsnorm(dests[j], w["norms"][j], cfg["eps"]) for j in range(3))
        a = static_conv(attention(cfg, w, xq, xk, xv, positions, world), w["attn_conv"])
        first = layer % B == 0
        if first:
            stack = stack + [partial]
            m_in = mix(stack, a, mlp_q[layer], H)
        else:
            m_in = mix(stack, f32(partial + a), mlp_q[layer], H)
        if messages is not None:
            m_in = blend(m_in, messages[3], w["ctrl"]["mlp"])
        x_moe = rmsnorm(m_in, w["mlp_norm"], cfg["eps"])
        visit_mlp_in.append(x_moe)
        y, route = moe(cfg, w, teacher_mlp_in[visit] if teacher_mlp_in is not None else x_moe, native, world)
        m = static_conv(y, w["mlp_conv"])
        partial = bf16(a + m) if first else bf16(partial + a + m)
        visit_attn.append(a)
        visit_mlp.append(m)
        routes.append(route)
        visit_states.append(partial.copy())
        if teacher is not None:
            partial = teacher[visit].copy()
    hn = rmsnorm(mix(stack, partial, out_q, H), load_np(entries, "model.llm.norm.weight"), cfg["eps"])
    head = load_np(entries, "model.llm.unembed.weight")[:cfg["vocab"]]
    logits = f32(hn @ head.T)
    return visit_states, visit_attn, visit_mlp, visit_mlp_in, routes, hn, logits


def topk_rows(logits, k):
    ids, vals = [], []
    for row in logits:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:k]
        ids.extend(order)
        vals.extend(float(row[i]) for i in order)
    return ids, vals


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-inkling2-reference-dump", "version": VERSION, "backend": "numpy",
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


def read_dump_tokens(path):
    """The token ids of another dump in this format."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != MAGIC:
        raise ValueError("%s is not a reference dump" % path)
    _, header_len = struct.unpack("<II", blob[8:16])
    t = json.loads(blob[16:16 + header_len])["tensors"]["tokens"]
    base = 16 + header_len + t["offset"]
    return list(struct.unpack("<%dq" % t["shape"][0], blob[base:base + t["nbytes"]]))


def gen_pure(args):
    cfg = text_config(args.checkpoint_dir)
    entries = read_index(args.checkpoint_dir)
    if args.tokens_from:
        tokens = read_dump_tokens(args.tokens_from)
        args.tokens = len(tokens)
    else:
        rng = make_rng(args.seed)
        tokens = [int(rng() * cfg["vocab"]) % cfg["vocab"] for _ in range(args.tokens)]
    native = args.activations == "native"
    teacher = load_engine_states(args.teacher) if args.teacher else None
    visits = visits_of(cfg)
    if teacher is not None and (len(teacher[0]) != len(visits) or teacher[0][0].shape != (args.tokens, cfg["hidden"])):
        raise ValueError("the engine states' shape is not this dump's")
    states, attn, mlp, mlp_in, routes, hn, logits = reference_forward(cfg, entries, tokens, native=native,
                                                                      teacher=teacher, progress=True, world=args.world)
    topk = 8
    top_ids, top_vals = topk_rows(logits, topk)
    capped = [cfg["logit_cap"] * float(np.tanh(v / cfg["logit_cap"])) for v in top_vals]
    T, H, V, K = args.tokens, cfg["hidden"], len(visits), cfg["top_k"]
    flat_routes = [e for visit in routes for row in visit for e in row]
    tensors = {
        "tokens": ("I64", [T], struct.pack("<%dq" % T, *tokens)),
        "final_hidden": ("BF16", [T, H], bf16_to_bytes(hn.reshape(-1))),
        "topk_ids": ("I32", [T, topk], struct.pack("<%di" % (T * topk), *top_ids)),
        "topk_logits": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *top_vals)),
        "topk_capped": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *capped)),
        "visit_states": ("BF16", [V, T, H], bf16_to_bytes(np.stack(states).reshape(-1))),
        "visit_attn": ("BF16", [V, T, H], bf16_to_bytes(np.stack(attn).reshape(-1))),
        "visit_mlp": ("BF16", [V, T, H], bf16_to_bytes(np.stack(mlp).reshape(-1))),
        "visit_mlp_in": ("BF16", [V, T, H], bf16_to_bytes(np.stack(mlp_in).reshape(-1))),
        "route_ids": ("I32", [V, T, K], struct.pack("<%di" % len(flat_routes), *flat_routes)),
    }
    cfg_summary = {"hidden": H, "vocab": cfg["vocab"], "layers": cfg["layers"], "visits": V, "tokens": T,
                   "top_k": topk, "activations": args.activations, "teacher": teacher is not None,
                   "world": args.world}
    write_dump(args.out, cfg_summary, tensors)
    print("wrote %s (%d tokens, %d visits, %s activations)" % (args.out, T, V, args.activations))
    return 0


def check_quant(args):
    """The engine's activation quantizers against this file's: the rows and
    results of tests/cuda/inkling2_kernels_test --dump-quant."""
    blob = open(args.dump, "rb").read()
    rows, k = struct.unpack("<ii", blob[:8])
    n = rows * k
    off = 8

    def take_bf16():
        nonlocal off
        bits = np.frombuffer(blob, dtype=np.uint16, count=n, offset=off).astype(np.uint32) << 16
        off += 2 * n
        return bits.view(np.float32).astype(np.float64).reshape(rows, k)

    x = take_bf16()
    inner = take_bf16()
    outer = np.frombuffer(blob, dtype=np.float32, count=rows, offset=off).copy()
    off += 4 * rows
    mx = take_bf16()
    bad_inner = bad_outer = bad_mx = 0
    for r in range(rows):
        want_inner, amax = nvfp4_quantize_row(x[r])
        want_outer = np.float32(amax * np.float32(1.0 / 1536.0))
        row_bad = int(np.count_nonzero(want_inner != inner[r]))
        if row_bad and bad_inner < 3 * k:
            c = int(np.flatnonzero(want_inner != inner[r])[0])
            print("row %d: %d inner values differ, first at %d: engine %r numpy %r (x %r, amax %r)" %
                  (r, row_bad, c, inner[r][c], want_inner[c], x[r][c], float(amax)))
        bad_inner += row_bad
        bad_outer += int(want_outer != outer[r])
        want_mx = mxfp8_quantize_row(x[r])
        row_bad = int(np.count_nonzero(want_mx != mx[r]))
        if row_bad and bad_mx < 3 * k:
            c = int(np.flatnonzero(want_mx != mx[r])[0])
            print("row %d: %d mxfp8 values differ, first at %d: engine %r numpy %r (x %r)" %
                  (r, row_bad, c, mx[r][c], want_mx[c], x[r][c]))
        bad_mx += row_bad
    print("quantizers against numpy over %d rows of %d: nvfp4 inner %d, nvfp4 outer %d, mxfp8 %d differing" %
          (rows, k, bad_inner, bad_outer, bad_mx))
    return 1 if bad_inner or bad_outer or bad_mx else 0


# ---------------------------------------------------------------------------
# the draft model (mtp.safetensors): docs/inkling2_mini_plan.md section 1.7,
# the released reference vLLM's models/inkling2/nvidia/dspark.py
# ---------------------------------------------------------------------------

DRAFT_MAGIC = b"DGPPINKD"


def draft_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        full = json.load(f)
    m = full["mtp_config"]
    tc = full["text_config"]
    local = set(m["local_layer_ids"])
    return {
        "hidden": m["hidden_size"], "layers": m["num_hidden_layers"], "heads": m["num_attention_heads"],
        "kv_heads": m["num_key_value_heads"], "head_dim": m["head_dim"], "inter": m["intermediate_size"],
        "eps": m.get("rms_norm_eps", 1e-6), "theta": m.get("rope_theta", 1e6),
        "local": [1 if i in local else 0 for i in range(m["num_hidden_layers"])],
        "window": m["sliding_window_size"], "block": m["block_size"], "rank": m["markov_rank"],
        "taps": list(m["target_layer_ids"]),
        "vocab": tc["unpadded_vocab_size"], "cap": tc["final_logit_softcapping"],
    }


def read_draft_index(checkpoint_dir):
    entries = {}
    _parse_shard_header(os.path.join(checkpoint_dir, "mtp.safetensors"), entries)
    return entries


def bf16_bits(raw):
    """uint16 bf16 bits as float64 values."""
    return (np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def draft_weights(dcfg, entries):
    w = {"layers": []}
    for l in range(dcfg["layers"]):
        p = "layers.%d." % l
        w["layers"].append({
            "input_norm": load_np(entries, p + "input_layernorm.weight"),
            "post_norm": load_np(entries, p + "post_attention_layernorm.weight"),
            "dflare_norm": load_np(entries, p + "dflare_norm.weight"),
            "q": load_np(entries, p + "self_attn.q_proj.weight"),
            "k": load_np(entries, p + "self_attn.k_proj.weight"),
            "v": load_np(entries, p + "self_attn.v_proj.weight"),
            "k_ctx": load_np(entries, p + "self_attn.k_proj_ctx.weight"),
            "v_ctx": load_np(entries, p + "self_attn.v_proj_ctx.weight"),
            "o": load_np(entries, p + "self_attn.o_proj.weight"),
            "q_norm": load_np(entries, p + "self_attn.q_norm.weight"),
            "k_norm": load_np(entries, p + "self_attn.k_norm.weight"),
            "k_norm_ctx": load_np(entries, p + "self_attn.k_norm_ctx.weight"),
            "gate": load_np(entries, p + "mlp.gate_proj.weight"),
            "up": load_np(entries, p + "mlp.up_proj.weight"),
            "down": load_np(entries, p + "mlp.down_proj.weight"),
        })
    w["norm"] = load_np(entries, "norm.weight")
    w["mask"] = load_np(entries, "mask_embed")
    w["feature_norm"] = load_np(entries, "markov_head.dflare_norm.weight")
    w["w1"] = load_np(entries, "markov_head.markov_w1.weight")
    w["rnn_w"] = load_np(entries, "markov_head.rnn_proj.weight")
    w["rnn_b"] = load_np(entries, "markov_head.rnn_proj.bias")
    w["out_w"] = load_np(entries, "markov_head.hidden_out.weight")
    w["out_b"] = load_np(entries, "markov_head.hidden_out.bias")
    # The mixes' weights: the float32 softmax of each row, stored as bf16.
    fuse = load_np(entries, "fuse_weight").astype(np.float32)
    e = np.exp(fuse - fuse.max(axis=1, keepdims=True), dtype=np.float32)
    w["alpha"] = bf16((e / e.sum(axis=1, keepdims=True, dtype=np.float32)).astype(np.float32))
    return w


def draft_rope(x, positions, dcfg):
    """Neox rotary pairs (i, i + D/2) of x [N, heads, D] at `positions`, the
    angles in float32 as the reference's table."""
    D = dcfg["head_dim"]
    half = D // 2
    inv = (1.0 / (np.float32(dcfg["theta"]) ** (np.arange(0, D, 2, dtype=np.float32) / np.float32(D)))).astype(np.float32)
    ang = (np.asarray(positions, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    cos = np.cos(ang).astype(np.float64)[:, None, :]
    sin = np.sin(ang).astype(np.float64)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return bf16(np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1))


def draft_head_norm_rope(x, w, positions, dcfg):
    """Per-head RMSNorm rounded to bf16, then the rotation."""
    n = bf16(x * rstd(x, dcfg["eps"]) * w)
    return draft_rope(n, positions, dcfg)


def draft_mix(features, alpha, norm_w, eps):
    """bf16(m * rstd(m) * w) with m the alpha-weighted sum of the taps.
    features [N, taps, H]."""
    m = f32(np.einsum("nth,t->nh", features, alpha))
    return bf16(m * rstd(m, eps) * norm_w)


def draft_context_kv(dcfg, dw, features, positions):
    """Per draft layer, the context keys [N, kv_heads, D] and values of the
    positions whose target features are `features`."""
    out = []
    Hkv, D = dcfg["kv_heads"], dcfg["head_dim"]
    for l, lw in enumerate(dw["layers"]):
        ctx = draft_mix(features, dw["alpha"][l], lw["dflare_norm"], dcfg["eps"])
        k = bf16(ctx @ lw["k_ctx"].T).reshape(-1, Hkv, D)
        v = bf16(ctx @ lw["v_ctx"].T).reshape(-1, Hkv, D)
        out.append((draft_head_norm_rope(k, lw["k_norm_ctx"], positions, dcfg), v))
    return out


def draft_block(dcfg, dw, anchor_embed, pos0, context):
    """The block's final rows [block, H]: the anchor's normed embedding and
    the mask rows through the layers. `context` is draft_context_kv's list
    over positions 0 .. pos0 - 1 (or the engine's own rows in that shape)."""
    B, H = dcfg["block"], dcfg["hidden"]
    Hq, Hkv, D = dcfg["heads"], dcfg["kv_heads"], dcfg["head_dim"]
    group = Hq // Hkv
    positions = pos0 + np.arange(B)
    resid = np.tile(dw["mask"], (B, 1))
    resid[0] = anchor_embed
    x = rmsnorm(resid, dw["layers"][0]["input_norm"], dcfg["eps"])
    for l, lw in enumerate(dw["layers"]):
        q = bf16(x @ lw["q"].T).reshape(B, Hq, D)
        k = bf16(x @ lw["k"].T).reshape(B, Hkv, D)
        v = bf16(x @ lw["v"].T).reshape(B, Hkv, D)
        q = draft_head_norm_rope(q, lw["q_norm"], positions, dcfg)
        k = draft_head_norm_rope(k, lw["k_norm"], positions, dcfg)
        # Every row of the block sees the same keys: all of them, or on a
        # sliding layer the last `window` up to the block's last row.
        lo = max(0, pos0 + B - dcfg["window"]) if dcfg["local"][l] else 0
        keys = np.concatenate([context[l][0][lo:pos0], k], axis=0)
        vals = np.concatenate([context[l][1][lo:pos0], v], axis=0)
        out = np.zeros((B, Hq, D))
        for h in range(Hq):
            logits = (q[:, h] @ keys[:, h // group].T) * D ** -0.5
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p = p / p.sum(axis=1, keepdims=True)
            out[:, h] = p @ vals[:, h // group]
        o = bf16(bf16(out.reshape(B, Hq * D)) @ lw["o"].T)
        s = f32(resid + o)
        resid = bf16(s)
        x = bf16(s * rstd(s, dcfg["eps"]) * lw["post_norm"])
        g = bf16(x @ lw["gate"].T)
        u = bf16(x @ lw["up"].T)
        act = bf16(g * sigmoid(g) * u)
        down = bf16(act @ lw["down"].T)
        s = f32(resid + down)
        resid = bf16(s)
        nxt = dw["layers"][l + 1]["input_norm"] if l + 1 < len(dw["layers"]) else dw["norm"]
        x = bf16(s * rstd(s, dcfg["eps"]) * nxt)
    return x


def draft_fold(dw, head):
    """The head rows through the recurrent head's output projection."""
    return bf16(head @ dw["out_w"]), bf16(head @ dw["out_b"])


def draft_head(dcfg, dw, hidden, feature, anchor, head, fold, forced=None):
    """The proposals of one block: per step the capped logits over the
    vocabulary and the pick (the first maximum). `forced` replaces the
    predecessor of every step after the first (the engine's own picks)."""
    R, B = dcfg["rank"], dcfg["block"]
    rows = hidden[1:]
    base = bf16(rows @ head.T)
    fixed = f32(dw["rnn_b"] + np.concatenate([rows, np.tile(feature, (B - 1, 1))], axis=1) @ dw["rnn_w"][:, 2 * R:].T)
    state = np.zeros(R)
    prev = anchor
    cap = dcfg["cap"]
    picks, rows_out = [], []
    for i in range(B - 1):
        raw = bf16(f32(fixed[i] + np.concatenate([state, dw["w1"][prev]]) @ dw["rnn_w"][:, :2 * R].T))
        gate = bf16(sigmoid(raw[:R]))
        cand = bf16(np.tanh(raw[R:2 * R]))
        state = bf16(bf16(gate * state) + bf16(bf16(1.0 - gate) * cand))
        latent = bf16(np.tanh(raw[2 * R:]))
        bias = bf16(f32(fold[1] + fold[0] @ latent))
        x = bf16(base[i] + bias)
        logits = bf16(cap * np.tanh(x / cap)) if cap > 0 else x
        pick = int(np.argmax(logits))
        picks.append(pick)
        rows_out.append(logits)
        prev = pick if forced is None else int(forced[i])
    return picks, rows_out


def check_draft(args):
    """The engine's draft model against this file's, from the dump of
    tests/cuda/inkling2_draft_test --dump: per case the tapped states of a
    prefill, the context keys and values the engine stored, its block's
    final rows and its proposals. Every stage is recomputed from the
    engine's own input to that stage."""
    cfg = text_config(args.checkpoint_dir)
    dcfg = draft_config(args.checkpoint_dir)
    entries = read_index(args.checkpoint_dir)
    dw = draft_weights(dcfg, read_draft_index(args.checkpoint_dir))
    embed = load_np(entries, "model.llm.embed.weight")
    embed_norm = load_np(entries, "model.llm.embed_norm.weight")
    head = load_np(entries, "model.llm.unembed.weight")[:cfg["vocab"]]
    fold = draft_fold(dw, head)
    blob = open(args.dump, "rb").read()
    if blob[:8] != DRAFT_MAGIC:
        raise ValueError("%s is not a draft dump" % args.dump)
    version, H, taps, L, kvw, B, cases = struct.unpack("<7i", blob[8:36])
    if (version, H, taps, L, B) != (1, dcfg["hidden"], len(dcfg["taps"]), dcfg["layers"], dcfg["block"]):
        raise ValueError("the dump's geometry is not this checkpoint's draft model")
    Hkv, D = dcfg["kv_heads"], dcfg["head_dim"]
    if kvw != Hkv * D:
        raise ValueError("the dump is one rank's slice: the reference checks world 1")
    off = 36
    failures = 0
    for case in range(cases):
        p, _ = struct.unpack("<ii", blob[off:off + 8])
        anchor, = struct.unpack("<q", blob[off + 8:off + 16])
        off += 16

        def take(count):
            nonlocal off
            out = bf16_bits(np.frombuffer(blob, dtype=np.uint16, count=count, offset=off))
            off += 2 * count
            return out

        features = take(taps * p * H).reshape(taps, p, H).transpose(1, 0, 2)
        engine_kv = take(L * 2 * p * kvw).reshape(L, 2, p, Hkv, D)
        engine_hidden = take(B * H).reshape(B, H)
        engine_picks = list(struct.unpack("<%di" % (B - 1), blob[off:off + 4 * (B - 1)]))
        off += 4 * (B - 1)

        # The context keys and values from the engine's tapped states.
        want_kv = draft_context_kv(dcfg, dw, features, np.arange(p))
        worst_kv = 0.0
        for l in range(L):
            for j, name in enumerate(("keys", "values")):
                want, got = want_kv[l][j], engine_kv[l, j]
                err = float(np.sqrt(np.sum((want - got) ** 2) / max(np.sum(want ** 2), 1e-30)))
                worst_kv = max(worst_kv, err)
                if err > 2e-3:
                    print("case %d (context %d): layer %d context %s relative l2 %.5f" % (case, p, l, name, err))
                    failures += 1
        # The block from the engine's own context rows.
        anchor_embed = rmsnorm(embed[anchor], embed_norm, cfg["eps"])
        context = [(engine_kv[l, 0], engine_kv[l, 1]) for l in range(L)]
        want_hidden = draft_block(dcfg, dw, anchor_embed, p, context)
        err_hidden = float(np.sqrt(np.sum((want_hidden - engine_hidden) ** 2) / max(np.sum(want_hidden ** 2), 1e-30)))
        if err_hidden > 2e-2:
            print("case %d (context %d): block rows relative l2 %.5f" % (case, p, err_hidden))
            failures += 1
        # The head from the engine's own block rows, each step after the
        # engine's own previous pick. A differing pick must be a near tie.
        feature = draft_mix(features[p - 1:p], dw["alpha"][L], dw["feature_norm"], dcfg["eps"])[0] if p > 0 \
            else np.zeros(H)
        picks, rows = draft_head(dcfg, dw, engine_hidden, feature, anchor, head, fold, forced=engine_picks)
        ties = 0
        for i, (want, got) in enumerate(zip(picks, engine_picks)):
            if want == got:
                continue
            gap = float(rows[i][want] - rows[i][got])
            # One bf16 step at the logits' magnitude.
            step = float(np.spacing(np.float32(abs(rows[i][want]) + 1e-30))) * 65536.0
            if gap <= 2.0 * step:
                ties += 1
            else:
                print("case %d (context %d): proposal %d is %d, the reference picks %d (logit gap %.4f)" %
                      (case, p, i, got, want, gap))
                failures += 1
        print("case %d: context %d, anchor %d: context keys and values within %.5f, block rows within %.5f, "
              "proposals %s (%d near ties)" % (case, p, anchor, worst_kv, err_hidden, engine_picks, ties))
    if off != len(blob):
        raise ValueError("the dump has %d bytes past its cases" % (len(blob) - off))
    print("draft model against numpy over %d cases: %d failure(s)" % (cases, failures))
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-pure")
    g.add_argument("--checkpoint-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tokens", type=int, default=72)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--tokens-from", help="take the token ids of another dump instead of drawing them")
    g.add_argument("--world", type=int, default=1, choices=(1, 2, 4),
                   help="the tensor-parallel world whose partial sums and activation slices to follow")
    g.add_argument("--activations", choices=("bf16", "native"), default="bf16")
    g.add_argument("--teacher", help="the engine's partial states per visit: every visit past the first is fed "
                   "the engine's state after the visit before it")
    g.set_defaults(func=gen_pure)
    q = sub.add_parser("check-quant")
    q.add_argument("--dump", required=True)
    q.set_defaults(func=check_quant)
    d = sub.add_parser("check-draft")
    d.add_argument("--checkpoint-dir", required=True)
    d.add_argument("--dump", required=True)
    d.set_defaults(func=check_draft)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
