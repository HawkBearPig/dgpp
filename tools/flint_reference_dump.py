#!/usr/bin/env python3
"""Flint full-model reference dump (docs/flint_plan.md G5, 2026-10-22).

Computes the FlintForCausalLM forward — the embedding, the pre-norm decoder
layers (the BF16 q / k / v projections, the per-head one-rounding q / k
RMSNorm, RoPE over the whole head on the RoPE layers, the sliding-window
GQA attention with its per-head sink on the sliding layers / full attention
on the others; the BF16 SwiGLU MLP on the dense layers, the sigmoid-routed
top-k MoE in the layer's per-projection fp8 / NVFP4 formats plus the
ungated BF16 shared expert on the others), the final norm, the lm head —
in numpy doubles with bf16 rounding at the engine's boundaries, from the
SAME checkpoint the engine runs, and writes the comparable outputs (tokens,
every layer's residual, the final read, per-token top-k logits, the routing
decisions) as a DGPPFLNT dump read by tests/cuda/flint_forward_test.cpp.

The rounding points are the engine's (src/models/flint/attn_reference.cpp,
src/models/glm/moe_reference.cpp, kernels/add_rmsnorm.cu's one-rounding
form): every norm rounds once — y = bf16(x x rstd x w); fp8 experts read
through the dequant bridge w = bf16(e4m3(code) x scale); NVFP4 experts
exact e2m1(code) x e4m3(scale) with the fp32 reciprocal global dividing
the finished dot (bf16(dot / g) on gate / up, dot / g unrounded on down);
the MoE chain: unrounded down dots, the experts in ascending id order,
the shared expert last with weight 1, one rounding of the sum.

Usage:
  flint_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE [--tokens T] [--seed S] [--teacher STATES]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_reference_dump import E2M1_TABLE, bf16, e4m3_decode  # noqa: E402
from glm_reference_dump import read_safetensors_index  # noqa: E402
from kda_reference_dump import make_rng  # noqa: E402

MAGIC = b"DGPPFLNT"
VERSION = 1

D = 128          # q / k / v head width
ATTN_TILE = 32   # kernels/sink_attn.hpp's tile: the prefill chain runs one split of 32-token tiles


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def bf16_to_bytes(values):
    a = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

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
    if dtype in ("F8_E4M3", "U8", "I8"):
        return np.frombuffer(buf, dtype=np.uint8).reshape(shape)
    raise ValueError("unsupported dtype %s for %s" % (dtype, name))


def load_fp8(entries, base):
    """An fp8 pair on the 128 x 128 grid -> the engine's dequant bridge:
    bf16(e4m3 x scale) as an exact [N, K] float64 matrix (divisor 1)."""
    p = load_np(entries, base + ".weight")
    s = load_np(entries, base + ".weight_scale_inv")
    n, k = p.shape
    sc = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[:n, :k]
    return bf16(e4m3_decode(p).astype(np.float32) * sc.astype(np.float32)), 1.0


def load_nvfp4(entries, base):
    """An NVFP4 triple (`.weight_packed` U8 pairs, `.weight_scale` e4m3 per 16,
    `.weight_global_scale` F32 [1]) -> the exact [N, K] floats e2m1 x e4m3
    and the global the kernels divide the finished dot by."""
    p = load_np(entries, base + ".weight_packed")
    s = load_np(entries, base + ".weight_scale")
    g = float(load_np(entries, base + ".weight_global_scale").reshape(-1)[0])
    n, half = p.shape
    codes = np.empty((n, half * 2), dtype=np.int64)
    codes[:, 0::2] = p & 0xF
    codes[:, 1::2] = p >> 4
    vals = E2M1_TABLE[codes]
    sc = np.repeat(e4m3_decode(s).astype(np.float64), 16, axis=1)[:, : half * 2]
    return vals * sc, g


def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        tc = json.load(f)
    eos = tc["eos_token_id"]
    layers = tc["num_hidden_layers"]
    swa = [1 if t == "sliding_attention" else 0 for t in tc["layer_types"]]
    rope = [1] * layers
    for l in tc.get("no_rope_layer_indices", []):
        rope[l] = 0
    moe = [1] * layers
    for l in tc.get("mlp_only_layers", []):
        moe[l] = 0
    pbl = tc["quantization_config"]["routed_experts"]["precision_by_layer"]
    fp4 = {}
    for l in range(layers):
        if moe[l]:
            e = pbl[str(l)]
            fp4[l] = tuple(e[w] == "nvfp4" for w in ("gate", "up", "down"))
    return {
        "hidden": tc["hidden_size"], "vocab": tc["vocab_size"], "num_layers": layers,
        "swa": swa, "rope": rope, "moe": moe, "fp4": fp4,
        "eps": tc.get("rms_norm_eps", 1e-6),
        "eos": eos[0] if isinstance(eos, list) else eos,
        "heads": tc["num_attention_heads"], "kv_heads": tc["num_key_value_heads"],
        "theta": float(tc.get("rope_theta", 1e4)),
        "window": int(tc.get("sliding_window", 0)),
        "dense_inter": tc["intermediate_size"], "moe_inter": tc["moe_intermediate_size"],
        "shared_inter": tc["shared_expert_intermediate_size"],
        "experts": tc["num_experts"], "top_k": tc["num_experts_per_tok"],
        "norm_topk": tc.get("norm_topk_prob", True),
    }


def load_engine_states(path):
    """The engine's layer residuals (DGPP_FLINT_DUMP_ENGINE_STATES: int32 L,
    T, H, then bf16 rows per layer, then the post-final-norm hidden): the
    teacher-forced mode feeds each reference layer the engine's own input."""
    with open(path, "rb") as f:
        L, T, H = struct.unpack("<iii", f.read(12))
        raw = np.frombuffer(f.read(L * T * H * 2), dtype=np.uint16).reshape(L, T, H)
        fin = np.frombuffer(f.read(T * H * 2), dtype=np.uint16).reshape(T, H)
    to_f = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return [to_f(raw[l]) for l in range(L)], to_f(fin)


def layer_weights(cfg, entries, layer):
    p = "model.layers.%d." % layer
    w = {"moe": cfg["moe"][layer] == 1, "swa": cfg["swa"][layer] == 1, "rope": cfg["rope"][layer] == 1}
    w["input_norm"] = load_np(entries, p + "input_layernorm.weight")
    w["post_norm"] = load_np(entries, p + "post_attention_layernorm.weight")
    a = p + "self_attn."
    w["q"] = load_np(entries, a + "q_proj.weight")  # [heads * D, H]
    w["k"] = load_np(entries, a + "k_proj.weight")  # [kv * D, H]
    w["v"] = load_np(entries, a + "v_proj.weight")
    w["o"] = load_np(entries, a + "o_proj.weight")  # [H, heads * D]
    w["q_norm"] = load_np(entries, a + "q_norm.weight")
    w["k_norm"] = load_np(entries, a + "k_norm.weight")
    w["sink"] = load_np(entries, a + "softmax_offset")
    m = p + "mlp."
    if w["moe"]:
        E = cfg["experts"]
        fp4 = cfg["fp4"][layer]
        w["router"] = load_np(entries, m + "gate.weight")
        w["shared"] = (load_np(entries, m + "shared_expert.gate_proj.weight"),
                       load_np(entries, m + "shared_expert.up_proj.weight"),
                       load_np(entries, m + "shared_expert.down_proj.weight"))
        experts = []
        for e in range(E):
            trip = []
            for i, name in enumerate(("gate_proj", "up_proj", "down_proj")):
                base = m + "experts.%d.%s" % (e, name)
                trip.append(load_nvfp4(entries, base) if fp4[i] else load_fp8(entries, base))
            experts.append(trip)
        w["experts"] = experts
    else:
        w["dense"] = (load_np(entries, m + "gate_proj.weight"), load_np(entries, m + "up_proj.weight"),
                      load_np(entries, m + "down_proj.weight"))
    return w


# ---------------------------------------------------------------------------
# modules (double interior, bf16 at the engine's boundaries)
# ---------------------------------------------------------------------------

def rmsnorm1(x, w, eps):
    """The one-rounding RMSNorm over the last axis: y = bf16(x * rstd * w),
    rstd = 1 / sqrt(float32(mean(x^2)) + eps) as the kernels narrow it."""
    x = np.asarray(x, dtype=np.float64)
    var = np.mean(x * x, axis=-1, keepdims=True).astype(np.float32).astype(np.float64)
    rstd = 1.0 / np.sqrt(var + np.float64(np.float32(eps)))
    return bf16(x * rstd * w)


def gemm_bf16(x, w):
    """bf16(x @ w^T): a bf16 Linear, one rounding (fp32 accumulate in the engine)."""
    return bf16(np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64).T)


def rope_inv(theta):
    """kernels/qsa.hpp's table over the whole head: inv[i] = theta^(-2i/128) in fp32."""
    base = np.float32(theta)
    return np.array([np.float32(1.0) / np.float32(base ** np.float32(2 * i / D)) for i in range(D // 2)],
                    dtype=np.float32).astype(np.float64)


def rope_heads(x, pos, inv):
    """x [heads, D] (bf16 values): rotate_half over the whole head — pairs
    (i, i + 64), fp32 cos / sin of fp32 pos x inv_freq, one rounding of each
    rotated element (attn_reference.cpp)."""
    ang = (np.float32(pos) * inv.astype(np.float32)).astype(np.float64)
    cos = np.cos(ang).astype(np.float32).astype(np.float64)
    sin = np.sin(ang).astype(np.float32).astype(np.float64)
    half = D // 2
    x1, x2 = x[:, :half], x[:, half:]
    r1 = bf16(x1 * cos - x2 * sin)
    r2 = bf16(x2 * cos + x1 * sin)
    return np.concatenate([r1, r2], axis=1)


def attention_forward(x_rows, w, cfg, T):
    """Rows t = 0 .. T-1 at positions t; row t attends [max(0, t - W + 1), t]
    on a sliding-window layer, [0, t] on a full one. The kernel's online
    chain over 32-token tiles from the first visible row: each tile's
    probabilities rounded to bf16 against the running max, the sums
    rescaled, the sink as split 0's opening (max, sum) = (sink, 1), the
    denominator unrounded, out = bf16(c / l)."""
    lh, kv = cfg["heads"], cfg["kv_heads"]
    hpk = lh // kv
    eps = cfg["eps"]
    inv = rope_inv(cfg["theta"]) if w["rope"] else None
    scale = float(np.float32(D ** -0.5))
    window = cfg["window"] if w["swa"] else 0
    x = np.asarray(x_rows, dtype=np.float64)
    q = bf16(x @ w["q"].T).reshape(T, lh, D)
    k = bf16(x @ w["k"].T).reshape(T, kv, D)
    v = bf16(x @ w["v"].T).reshape(T, kv, D)
    # The per-head one-rounding norms, then RoPE (or none).
    q = rmsnorm1(q, w["q_norm"], eps)
    k = rmsnorm1(k, w["k_norm"], eps)
    if inv is not None:
        for t in range(T):
            q[t] = rope_heads(q[t], t, inv)
            k[t] = rope_heads(k[t], t, inv)
    sink = w["sink"]
    out = np.empty((T, lh * D))
    for t in range(T):
        lo = max(0, t - window + 1) if window > 0 else 0
        n = t - lo + 1
        for h in range(lh):
            kvh = h // hpk
            s = (k[lo:t + 1, kvh, :] @ q[t, h, :]) * scale  # [n]
            m, l = float(sink[h]), 1.0
            c = np.zeros(D)
            for t0 in range(0, n, ATTN_TILE):
                t1 = min(n, t0 + ATTN_TILE)
                m_new = max(m, float(np.max(s[t0:t1])))
                rescale = np.exp(m - m_new)
                l = l * rescale + float(np.sum(np.exp(s[t0:t1] - m_new)))
                c *= rescale
                p = bf16(np.exp(s[t0:t1] - m_new))
                c += p @ v[lo + t0:lo + t1, kvh, :]
                m = m_new
            out[t, h * D:(h + 1) * D] = bf16(c / l)
    return gemm_bf16(out, w["o"])


def swiglu(g, u):
    """act = bf16(bf16(silu(g)) * u) — no clamps (FlintMLP)."""
    return bf16(bf16(g * sigmoid(g)) * u)


def dense_forward(x, dense):
    wg, wu, wd = dense
    return gemm_bf16(swiglu(gemm_bf16(x, wg), gemm_bf16(x, wu)), wd)


NEAR_TIES = []  # every routed row's margin between the K-th and (K+1)-th score


def moe_forward(x, w, cfg):
    """The router (sigmoid scores, no bias, ties to the lower id, the picked
    scores normalized) and the engine's chain: unrounded down dots, the
    experts in ascending id order, the shared expert (weight 1) last, one
    rounding. x: one row."""
    E, K = cfg["experts"], cfg["top_k"]
    scores = sigmoid(w["router"] @ x)
    order = sorted(range(E), key=lambda e: (-scores[e], e))
    sel = sorted(order[:K])
    NEAR_TIES.append(scores[order[K - 1]] - scores[order[K]])
    denom = (float(np.sum(scores[sel])) + 1e-20) if cfg["norm_topk"] else 1.0
    weights = [scores[e] / denom for e in sel]
    acc = np.zeros(cfg["hidden"])
    for e, we in zip(sel, weights):
        (wg, gg), (wu, gu), (wd, gd) = w["experts"][e]
        g = bf16((wg @ x) / gg)
        u = bf16((wu @ x) / gu)
        act = swiglu(g, u)
        acc += we * ((wd @ act) / gd)
    sg, su, sd = w["shared"]
    act = swiglu(bf16(sg @ x), bf16(su @ x))
    acc += sd @ act
    return bf16(acc), sel


def layer_forward(h_rows, w, cfg, T):
    """h += attn(input_norm(h)); h += mlp(post_norm(h)) — bf16 residual adds."""
    eps = cfg["eps"]
    x_rows = rmsnorm1(h_rows, w["input_norm"], eps)
    y = attention_forward(x_rows, w, cfg, T)
    h_rows = bf16(h_rows + y)
    x_rows = rmsnorm1(h_rows, w["post_norm"], eps)
    routes = []
    if w["moe"]:
        y = np.empty_like(h_rows)
        for t in range(T):
            yy, sel = moe_forward(x_rows[t], w, cfg)
            y[t] = yy
            routes.append(sel)
    else:
        y = dense_forward(x_rows, w["dense"])
    h_rows = bf16(h_rows + y)
    return h_rows, routes


def reference_forward(cfg, entries, tokens, progress=False, teacher=None):
    T = len(tokens)
    embed = load_np(entries, "model.embed_tokens.weight")
    h = embed[np.asarray(tokens)]
    layer_states, routes = [], []
    for layer in range(cfg["num_layers"]):
        w = layer_weights(cfg, entries, layer)
        if progress:
            print("layer %d (%s%s, %s)" % (layer, "swa" if w["swa"] else "full", "" if w["rope"] else " no-rope",
                                          "moe" if w["moe"] else "dense"), file=sys.stderr, flush=True)
        NEAR_TIES.clear()
        if teacher is not None and layer > 0:
            h = teacher[0][layer - 1]
        h, route = layer_forward(h, w, cfg, T)
        layer_states.append(h.copy())
        if w["moe"]:
            routes.append(route)
            if progress:
                tight = sorted((m, t) for t, m in enumerate(NEAR_TIES))[:3]
                print("  routing margins (smallest):" + "".join(" t%d %.3g" % (t, m) for m, t in tight),
                      file=sys.stderr, flush=True)
    if teacher is not None:
        h = teacher[0][cfg["num_layers"] - 1]
    final_norm = load_np(entries, "model.norm.weight")
    hn = rmsnorm1(h, final_norm, cfg["eps"])
    lm = load_np(entries, "lm_head.weight")
    logits = f32(hn @ lm.T)
    return layer_states, hn, logits, routes


def topk_rows(logits, k):
    ids, vals = [], []
    for row in logits:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:k]
        ids.extend(order)
        vals.extend(float(row[i]) for i in order)
    return ids, vals


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-flint-reference-dump", "version": VERSION, "backend": "numpy",
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


def gen_pure(args):
    cfg = text_config(args.checkpoint_dir)
    entries = read_safetensors_index(args.checkpoint_dir)
    rng = make_rng(args.seed)
    tokens = [int(rng() * cfg["vocab"]) % cfg["vocab"] for _ in range(args.tokens)]
    if args.tokens > 9:
        tokens[9] = cfg["eos"]
    teacher = load_engine_states(args.teacher) if args.teacher else None
    if teacher is not None and (len(teacher[0]) != cfg["num_layers"] or teacher[1].shape != (args.tokens, cfg["hidden"])):
        raise ValueError("the engine states' shape is not this dump's")
    layer_states, h_rows, logits, routes = reference_forward(cfg, entries, tokens, progress=True, teacher=teacher)
    topk = 8
    top_ids, top_vals = topk_rows(logits, topk)
    T, H = args.tokens, cfg["hidden"]
    L, K = cfg["num_layers"], cfg["top_k"]
    Lm = sum(cfg["moe"])
    flat_routes = [e for layer in routes for row in layer for e in row]
    tensors = {
        "tokens": ("I64", [T], struct.pack("<%dq" % T, *tokens)),
        "final_hidden": ("BF16", [T, H], bf16_to_bytes(h_rows.reshape(-1))),
        "topk_ids": ("I32", [T, topk], struct.pack("<%di" % (T * topk), *top_ids)),
        "topk_logits": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *top_vals)),
        "layer_states": ("BF16", [L, T, H], bf16_to_bytes(np.stack(layer_states).reshape(-1))),
        "route_ids": ("I32", [Lm, T, K], struct.pack("<%di" % len(flat_routes), *flat_routes)),
    }
    cfg_summary = {"hidden": H, "vocab": cfg["vocab"], "num_layers": L, "tokens": T, "top_k": topk, "moe_layers": Lm,
                   "teacher": teacher is not None}
    write_dump(args.out, cfg_summary, tensors)
    print("wrote %s (%d tokens, %d layers)" % (args.out, T, L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-pure")
    g.add_argument("--checkpoint-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tokens", type=int, default=72)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--teacher", help="the engine's layer states (DGPP_FLINT_DUMP_ENGINE_STATES): every layer "
                   "past the first is fed the engine's residual after the layer before it")
    g.set_defaults(func=gen_pure)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
