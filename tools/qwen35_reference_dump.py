#!/usr/bin/env python3
"""Reference forward for the Qwen3.8-27B family (qwen3_5; #84, 2026-10-03).

Computes the Qwen3_5ForConditionalGeneration text forward — the embedding,
the pre-norm decoder layers (Gated DeltaNet layers: the fp8 qkv / z
projections, the causal depthwise conv + silu, the L2-normalized q / k, the
gated delta recurrence with exp(A_log) and dt_bias, the swish-gated RMSNorm,
the out projection; full-attention layers: the fp8 [q | gate] / k / v
projections, the (1 + w) q / k norms, rotate-half RoPE over the first 64 of
256 dims, GQA causal attention in the kernel's 32-token tile chain, the
sigmoid output gate, the o projection; the fp8 SwiGLU MLP), the final norm,
the bf16 lm head — and the MTP draft block's rows over the same prompt
(pre_fc_norm_embedding / pre_fc_norm_hidden on the shifted embeddings and
the post-final-norm hidden, cat -> fc, the draft's full-attention layer,
mtp.norm, the shared head), in numpy doubles with bf16 rounding at the
engine's boundaries, from the SAME checkpoint the engine runs, and writes
the comparable outputs (tokens, every layer's residual, the final read,
per-token top-k logits, the draft's rows) as a DGPPQW35 dump read by
tests/cuda/qwen35_forward_test.cpp.

The rounding points are the engine's: kernels/qwen_norm (the (1 + w) norm
with one rounding, the gated norm's three), kernels/qsa (norm + rope: bf16
cos / sin, bf16 products, one rounding of the sum; the gate: bf16(bf16(c) x
bf16(sigmoid(g)))), kernels/full_attn (probabilities rounded to bf16 per
32-token tile against the running max, the denominator unrounded, one
rounding of c / l), models/qwen/model35_kernels (swiglu: bf16(silu(g) x u)),
the dequant bridge / fp8 GEMVs (w = bf16(e4m3(code) x scale), one fp32
product, one bf16 rounding), every GEMM's bf16 output rounded once from an
fp32 accumulation (here: exact doubles, rounded once).

Usage:
  qwen35_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE [--tokens T] [--seed S] [--teacher STATES]
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
from dsv41_reference_dump import bf16, e4m3_decode  # noqa: E402
from glm_reference_dump import read_safetensors_index  # noqa: E402
from kda_reference_dump import make_rng  # noqa: E402

MAGIC = b"DGPPQW35"
VERSION = 1
ATTN_TILE = 32  # kernels/full_attn.cu's kKTile: the prefill tile form's K/V tiles


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def bf16_to_bytes(values):
    a = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def load_np(entries, name):
    """BF16 / F32 as float64 (exact), F8_E4M3 as uint8 codes."""
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
    """One quantized dense matrix -> the engine's weights as an exact
    [N, K] float64 matrix. All four forms the qwen3_5 loader binds: the
    block-FP8 pair (BF16 weight_scale_inv on the 128 x 128 grid), the FP8
    release's BF16 head, and the mixed release's channel-FP8 pair (BF16
    weight_scale [N, 1]) and NVFP4 triple (weight_packed / e4m3
    weight_scale / F32 weight_global_scale, folded into the values here —
    the engine folds the global once at the end of each dot, which is the
    same float64 arithmetic)."""
    if base + ".weight_packed" in entries:
        return load_nvfp4(entries, base)
    if base + ".weight_scale_2" in entries:
        return load_nvfp4_modelopt(entries, base)
    p = load_np(entries, base + ".weight")
    if p.dtype == np.float64:  # a BF16 matrix (the FP8 release's head, the draft layer)
        return p
    if base + ".weight_scale_inv" in entries:
        s = load_np(entries, base + ".weight_scale_inv")
        n, k = p.shape
        sc = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[:n, :k]
        prod = e4m3_decode(p).astype(np.float32) * sc.astype(np.float32)
        return bf16(prod)
    s = load_np(entries, base + ".weight_scale")  # BF16 [N, 1]: one scale a row
    prod = e4m3_decode(p).astype(np.float32) * s.astype(np.float32)
    return bf16(prod)


_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def load_nvfp4(entries, base):
    """A compressed-tensors nvfp4-pack matrix: two e2m1 codes a byte (the
    LOW nibble the even element), one e4m3 scale per 16, divided once by
    the F32 weight global — the engine folds the same divisor at the end
    of each dot (the kernels divide by the slot), which is the same
    float64 arithmetic. The input global is the W4A4 activation-side
    factor and has no subject under exact activations. Direction pinned
    against the FP8 sibling release's per-layer RMS (2026-10-06)."""
    packed = load_np(entries, base + ".weight_packed").astype(np.uint8)
    n, kh = packed.shape
    codes = np.empty((n, kh * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    vals = _E2M1[codes & 7] * np.where(codes & 8, -1.0, 1.0)
    s = e4m3_decode(load_np(entries, base + ".weight_scale"))
    g = float(load_np(entries, base + ".weight_global_scale").reshape(-1)[0])
    return (vals * np.repeat(s, 16, axis=1)) / g


def load_nvfp4_modelopt(entries, base):
    """A ModelOpt NVFP4 matrix (nvidia 122B routed experts): two e2m1 codes
    a byte (LOW nibble even), one e4m3 scale per 16, TIMES the F32
    weight_scale_2 — the loader stores its reciprocal and the kernels
    divide once, which is the same multiply. input_scale has no subject
    under exact activations."""
    packed = load_np(entries, base + ".weight").astype(np.uint8)
    n, kh = packed.shape
    codes = np.empty((n, kh * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    vals = _E2M1[codes & 7] * np.where(codes & 8, -1.0, 1.0)
    s = e4m3_decode(load_np(entries, base + ".weight_scale"))
    g = float(load_np(entries, base + ".weight_scale_2").reshape(-1)[0])
    return (vals * np.repeat(s, 16, axis=1)) * g


def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        root = json.load(f)
    tc = root["text_config"]
    eos = tc.get("eos_token_id", 0)
    hd = int(tc["head_dim"])
    rp = tc.get("rope_parameters", {})
    is_moe = tc.get("model_type", "") == "qwen3_5_moe_text"
    out = {
        "hidden": int(tc["hidden_size"]), "vocab": int(tc["vocab_size"]), "num_layers": int(tc["num_hidden_layers"]),
        "layer_types": list(tc["layer_types"]), "eps": float(tc.get("rms_norm_eps", 1e-6)),
        "eos": eos[0] if isinstance(eos, list) else int(eos),
        "heads": int(tc["num_attention_heads"]), "kv_heads": int(tc["num_key_value_heads"]), "head_dim": hd,
        "rotary": int(round(hd * float(rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0))))),
        "theta": float(rp.get("rope_theta", tc.get("rope_theta", 1e7))),
        "gdn_kh": int(tc["linear_num_key_heads"]), "gdn_vh": int(tc["linear_num_value_heads"]),
        "gdn_kd": int(tc["linear_key_head_dim"]), "gdn_vd": int(tc["linear_value_head_dim"]),
        "gdn_conv": int(tc["linear_conv_kernel_dim"]),
        "inter": int(tc["moe_intermediate_size"]) if is_moe else int(tc["intermediate_size"]),
        "mtp": int(tc.get("mtp_num_hidden_layers", 0)),
        "is_moe": is_moe,
    }
    if is_moe:
        out.update({
            "num_experts": int(tc["num_experts"]), "top_k": int(tc["num_experts_per_tok"]),
            "shared_inter": int(tc["shared_expert_intermediate_size"]),
            "norm_topk": bool(tc.get("norm_topk_prob", True)),
        })
    return out


def load_engine_states(path):
    """The engine's layer residuals (qwen35_forward_test --engine-states:
    int32 L, T, H, then bf16 rows per layer, then the post-final-norm
    hidden): the teacher-forced mode feeds each reference layer the engine's
    own input, so one layer's error cannot cascade through the stack."""
    with open(path, "rb") as f:
        L, T, H = struct.unpack("<iii", f.read(12))
        raw = np.frombuffer(f.read(L * T * H * 2), dtype=np.uint16).reshape(L, T, H)
        fin = np.frombuffer(f.read(T * H * 2), dtype=np.uint16).reshape(T, H)
    to_f = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return [to_f(raw[l]) for l in range(L)], to_f(fin)


def layer_prefix(cfg, layer):
    return "mtp.layers.0." if layer == cfg["num_layers"] else "model.language_model.layers.%d." % layer


def layer_weights(cfg, entries, layer):
    p = layer_prefix(cfg, layer)
    kind = "full_attention" if layer == cfg["num_layers"] else cfg["layer_types"][layer]
    w = {"kind": kind, "input_norm": load_np(entries, p + "input_layernorm.weight"),
         "post_norm": load_np(entries, p + "post_attention_layernorm.weight")}
    if kind == "linear_attention":
        q = p + "linear_attn."
        w["gdn"] = {
            "qkv": load_fp8(entries, q + "in_proj_qkv"), "z": load_fp8(entries, q + "in_proj_z"),
            "a": load_np(entries, q + "in_proj_a.weight"), "b": load_np(entries, q + "in_proj_b.weight"),
            "conv": load_np(entries, q + "conv1d.weight")[:, 0, :],  # [C, CW]
            "a_log": load_np(entries, q + "A_log"), "dt_bias": load_np(entries, q + "dt_bias"),
            "norm": load_np(entries, q + "norm.weight"), "out": load_fp8(entries, q + "out_proj"),
        }
    else:
        q = p + "self_attn."
        w["attn"] = {
            "q": load_fp8(entries, q + "q_proj"), "k": load_fp8(entries, q + "k_proj"),
            "v": load_fp8(entries, q + "v_proj"), "o": load_fp8(entries, q + "o_proj"),
            "q_norm": load_np(entries, q + "q_norm.weight"), "k_norm": load_np(entries, q + "k_norm.weight"),
        }
    m = p + "mlp."
    if cfg.get("is_moe"):
        # Router + shared BF16, routed experts in the checkpoint's own form
        # (ModelOpt NVFP4 backbone, BF16 MTP draft): load_fp8 routes each.
        me = {"gate": load_np(entries, m + "gate.weight"),
              "shared_gate": load_np(entries, m + "shared_expert_gate.weight").reshape(-1),
              "shared": (load_np(entries, m + "shared_expert.gate_proj.weight"),
                         load_np(entries, m + "shared_expert.up_proj.weight"),
                         load_np(entries, m + "shared_expert.down_proj.weight")),
              "experts": []}
        for e in range(cfg["num_experts"]):
            ep = m + "experts.%d." % e
            me["experts"].append((load_fp8(entries, ep + "gate_proj"), load_fp8(entries, ep + "up_proj"),
                                  load_fp8(entries, ep + "down_proj")))
        w["mlp"] = me
    else:
        w["mlp"] = (load_fp8(entries, m + "gate_proj"), load_fp8(entries, m + "up_proj"), load_fp8(entries, m + "down_proj"))
    return w


# ---------------------------------------------------------------------------
# modules (double interior, bf16 at the engine's boundaries)
# ---------------------------------------------------------------------------
def rmsnorm(x, w, eps):
    """The zero-centered (1 + w) RMSNorm, one rounding (kernels/qwen_norm)."""
    x = np.asarray(x, dtype=np.float64)
    rstd = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return bf16(x * rstd * (1.0 + w))


def gemm(x, w):
    """bf16(x @ w^T): a bf16 Linear, the fp32 accumulation rounded once."""
    return bf16(np.asarray(x, dtype=np.float64) @ w.T)


def gdn_forward(x_rows, g, cfg, T):
    """The Gated DeltaNet layer over T rows from a zero state (the fp8
    projections, the causal conv + silu, the recurrence, the swish-gated
    norm, the out projection); returns [T, H]."""
    kh, vh, K, V = cfg["gdn_kh"], cfg["gdn_vh"], cfg["gdn_kd"], cfg["gdn_vd"]
    CW = cfg["gdn_conv"]
    C = 2 * kh * K + vh * V
    ratio = vh // kh
    scale = float(np.float32(K ** -0.5))
    qkv = gemm(x_rows, g["qkv"])  # [T, C]
    # The causal depthwise conv (zero state) + silu, rounded to bf16: tap
    # CW-1 multiplies the current input, tap j the input CW-1-j steps back.
    conv = np.zeros((T, C))
    for t in range(T):
        acc = np.zeros(C)
        for j in range(CW):
            src = t - (CW - 1) + j
            if src >= 0:
                acc += g["conv"][:, j] * qkv[src]
        conv[t] = bf16(acc / (1.0 + np.exp(-acc)))
    z = gemm(x_rows, g["z"])  # [T, vh * V]
    a = gemm(x_rows, g["a"])  # [T, vh]
    b = gemm(x_rows, g["b"])  # [T, vh]
    core = np.zeros((T, vh * V))
    A = np.exp(g["a_log"])
    for h in range(vh):
        hk = h // ratio
        S = np.zeros((V, K))
        for t in range(T):
            q = conv[t, hk * K:(hk + 1) * K]
            k = conv[t, kh * K + hk * K:kh * K + (hk + 1) * K]
            v = conv[t, 2 * kh * K + h * V:2 * kh * K + (h + 1) * V]
            qq = q / math.sqrt(float(np.dot(q, q)) + 1e-6) * scale
            kq = k / math.sqrt(float(np.dot(k, k)) + 1e-6)
            decay = math.exp(-(A[h] * float(softplus(a[t, h] + g["dt_bias"][h]))))
            beta = float(sigmoid(b[t, h]))
            d = (S @ kq) * decay
            u = (v - d) * beta
            S = S * decay + np.outer(u, kq)
            core[t, h * V:(h + 1) * V] = bf16(S @ qq)
    # The swish-gated RMSNorm per head: three roundings (kernels/qwen_norm).
    normed = np.zeros_like(core)
    for h in range(vh):
        xs = core[:, h * V:(h + 1) * V]
        zs = z[:, h * V:(h + 1) * V]
        rstd = 1.0 / np.sqrt(np.mean(xs * xs, axis=-1, keepdims=True) + cfg["eps"])
        uu = bf16(xs * rstd)
        pw = bf16(uu * g["norm"])
        normed[:, h * V:(h + 1) * V] = bf16(pw * (zs * sigmoid(zs)))
    return gemm(normed, g["out"])


def rope_inv(cfg):
    R = cfg["rotary"]
    return np.array([float(np.float32(1.0 / (np.float32(cfg["theta"]) ** np.float32(2 * i / R)))) for i in range(R // 2)])


def norm_rope(x, w, pos, inv, R, eps):
    """x [T, heads, D]: the (1 + w) norm per head, then rotate-half RoPE over
    the first R dims at position pos[t] with bf16 cos / sin, bf16 products
    and one rounding of the sum (kernels/qsa's norm_rope)."""
    T, heads, D = x.shape
    xn = rmsnorm(x, w, eps)
    out = xn.copy()
    half = R // 2
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]  # [T, half]
    c = bf16(np.cos(f32(ang)))
    s = bf16(np.sin(f32(ang)))
    c = c[:, None, :]
    s = s[:, None, :]
    lo, hi = xn[:, :, :half], xn[:, :, half:R]
    out[:, :, :half] = bf16(bf16(lo * c) + bf16(-hi * s))
    out[:, :, half:R] = bf16(bf16(hi * c) + bf16(lo * s))
    return out


def attention_forward(x_rows, w, cfg, T):
    """Rows t = 0 .. T-1 at positions t, row t attending [0, t]: the [q | gate]
    projection, the normed / roped q and k, the kernel's online chain over
    32-token tiles (each tile's probabilities rounded to bf16 against the
    running max, the sums rescaled, the denominator unrounded, c / l rounded
    once), the sigmoid output gate, the o projection."""
    qh, kvh, D, R = cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["rotary"]
    hpk = qh // kvh
    inv = rope_inv(cfg)
    pos = np.arange(T)
    qg = gemm(x_rows, w["q"]).reshape(T, qh, 2 * D)
    q = norm_rope(qg[:, :, :D], w["q_norm"], pos, inv, R, cfg["eps"])
    gate = qg[:, :, D:]
    k = norm_rope(gemm(x_rows, w["k"]).reshape(T, kvh, D), w["k_norm"], pos, inv, R, cfg["eps"])
    v = gemm(x_rows, w["v"]).reshape(T, kvh, D)
    scale = float(np.float32(D ** -0.5))
    out = np.empty((T, qh * D))
    for t in range(T):
        n = t + 1
        for h in range(qh):
            kv = h // hpk
            s = (k[:n, kv, :] @ q[t, h, :]) * scale
            m, l = -np.inf, 0.0
            c = np.zeros(D)
            for t0 in range(0, n, ATTN_TILE):
                t1 = min(n, t0 + ATTN_TILE)
                m_new = max(m, float(np.max(s[t0:t1])))
                rescale = math.exp(m - m_new) if m != -np.inf else 0.0
                l = l * rescale + float(np.sum(np.exp(s[t0:t1] - m_new)))
                c *= rescale
                p = bf16(np.exp(s[t0:t1] - m_new))
                c += p @ v[t0:t1, kv, :]
                m = m_new
            attn = bf16(c / l)
            out[t, h * D:(h + 1) * D] = bf16(attn * bf16(sigmoid(gate[t, h, :])))
    return gemm(out, w["o"])


def dense_forward(x, mlp):
    wg, wu, wd = mlp
    g = gemm(x, wg)
    u = gemm(x, wu)
    act = bf16(g / (1.0 + np.exp(-g)) * u)  # bf16(silu(g) x u): one rounding (qwen35_swiglu)
    return gemm(act, wd)


def moe_forward(x, me, cfg):
    """The routed MoE: bf16-rounded router logits (the kernel leaves the
    bf16 LOGIT in scores), softmax, top-k (+ renorm), ascending-fp32
    expert chain (silu(gate) x up, bf16 act like the dense path, down in
    fp64), shared sigmoid tail last, one bf16 rounding (QwenMoeLayer)."""
    logits = bf16(np.asarray(x, dtype=np.float64) @ me["gate"].T)
    mx = np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits - mx)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    T = x.shape[0]
    acc = np.zeros((T, cfg["hidden"]), dtype=np.float64)
    for t in range(T):
        order = sorted(range(cfg["num_experts"]), key=lambda e: (-probs[t, e], e))[:cfg["top_k"]]
        wt = probs[t, order]
        if cfg["norm_topk"]:
            wt = wt / np.sum(wt)
        row = np.zeros(cfg["hidden"], dtype=np.float64)
        for e in sorted(order):  # ascending expert order, like the engine chain
            wg, wu, wd = me["experts"][e]
            g = x[t] @ wg.T
            u = x[t] @ wu.T
            act = bf16(g / (1.0 + np.exp(-g)) * u)
            row += float(wt[list(order).index(e)]) * (act @ wd.T)
        sg, su, sd = me["shared"]
        s = float(sigmoid(x[t] @ me["shared_gate"]))
        gs = bf16((x[t] @ sg.T) / (1.0 + np.exp(-(x[t] @ sg.T))) * (x[t] @ su.T))
        row += s * (gs @ sd.T)
        acc[t] = row
    return bf16(acc)


def layer_forward(h_rows, w, cfg, T):
    """h += attn(input_norm(h)); h += mlp(post_norm(h)) — bf16 residual adds."""
    eps = cfg["eps"]
    x = rmsnorm(h_rows, w["input_norm"], eps)
    y = gdn_forward(x, w["gdn"], cfg, T) if w["kind"] == "linear_attention" else attention_forward(x, w["attn"], cfg, T)
    h_rows = bf16(h_rows + y)
    x = rmsnorm(h_rows, w["post_norm"], eps)
    mlp = w["mlp"]
    return bf16(h_rows + (moe_forward(x, mlp, cfg) if cfg.get("is_moe") else dense_forward(x, mlp)))


def reference_forward(cfg, entries, tokens, progress=False, teacher=None):
    """teacher: (layer states, final hidden) of the engine — every layer past
    the first takes the engine's residual after the layer before it, the
    final norm the engine's last state, so each layer is judged alone."""
    T = len(tokens)
    embed = load_np(entries, "model.language_model.embed_tokens.weight")
    h = embed[np.asarray(tokens)]
    layer_states = []
    for layer in range(cfg["num_layers"]):
        w = layer_weights(cfg, entries, layer)
        if progress:
            print("layer %d (%s)" % (layer, w["kind"]), file=sys.stderr, flush=True)
        if teacher is not None and layer > 0:
            h = teacher[0][layer - 1]
        h = layer_forward(h, w, cfg, T)
        layer_states.append(h.copy())
    if teacher is not None:
        h = teacher[0][cfg["num_layers"] - 1]
    hn = rmsnorm(h, load_np(entries, "model.language_model.norm.weight"), cfg["eps"])
    lm = load_fp8(entries, "lm_head")
    logits = f32(hn @ lm.T)
    return layer_states, hn, logits


def mtp_forward(cfg, entries, tokens, hn_main, progress=False):
    """The draft block over the prompt: row q embeds tokens[q + 1] and takes
    the main stack's post-final-norm hidden at q — cat([pre_fc_norm_embedding(e),
    pre_fc_norm_hidden(h_q)]) -> fc — then the draft's full-attention layer
    (positions q), mtp.norm and the shared lm head; (hidden, logits) for the
    T-1 rows."""
    eps = cfg["eps"]
    R = len(tokens) - 1
    embed = load_np(entries, "model.language_model.embed_tokens.weight")
    if progress:
        print("draft layer", file=sys.stderr, flush=True)
    e = rmsnorm(embed[np.asarray(tokens[1:])], load_np(entries, "mtp.pre_fc_norm_embedding.weight"), eps)
    hn = rmsnorm(hn_main[:R], load_np(entries, "mtp.pre_fc_norm_hidden.weight"), eps)
    x = gemm(np.concatenate([e, hn], axis=1), load_np(entries, "mtp.fc.weight"))
    w = layer_weights(cfg, entries, cfg["num_layers"])
    h = layer_forward(x, w, cfg, R)
    hn_out = rmsnorm(h, load_np(entries, "mtp.norm.weight"), eps)
    lm = load_fp8(entries, "lm_head")
    return hn_out, f32(hn_out @ lm.T)


def topk_rows(logits, k):
    ids, vals = [], []
    for row in logits:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:k]
        ids.extend(order)
        vals.extend(float(row[i]) for i in order)
    return ids, vals


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-qwen35-reference-dump", "version": VERSION, "backend": "numpy",
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
    layer_states, hn, logits = reference_forward(cfg, entries, tokens, progress=True, teacher=teacher)
    topk = 8
    top_ids, top_vals = topk_rows(logits, topk)
    T, H, L = args.tokens, cfg["hidden"], cfg["num_layers"]
    tensors = {
        "tokens": ("I64", [T], struct.pack("<%dq" % T, *tokens)),
        "final_hidden": ("BF16", [T, H], bf16_to_bytes(hn.reshape(-1))),
        "topk_ids": ("I32", [T, topk], struct.pack("<%di" % (T * topk), *top_ids)),
        "topk_logits": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *top_vals)),
        "layer_states": ("BF16", [L, T, H], bf16_to_bytes(np.stack(layer_states).reshape(-1))),
    }
    cfg_summary = {"hidden": H, "vocab": cfg["vocab"], "num_layers": L, "tokens": T, "top_k": topk,
                   "teacher": teacher is not None}
    if args.mtp and cfg["mtp"] >= 1 and T >= 2:
        mh, mlogits = mtp_forward(cfg, entries, tokens, teacher[1] if teacher is not None else hn, progress=True)
        m_ids, m_vals = topk_rows(mlogits, topk)
        tensors["mtp_final_hidden"] = ("BF16", [T - 1, H], bf16_to_bytes(mh.reshape(-1)))
        tensors["mtp_topk_ids"] = ("I32", [T - 1, topk], struct.pack("<%di" % ((T - 1) * topk), *m_ids))
        tensors["mtp_topk_logits"] = ("F32", [T - 1, topk], struct.pack("<%df" % ((T - 1) * topk), *m_vals))
        cfg_summary["mtp_rows"] = T - 1
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
    g.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True,
                   help="also dump the draft block's rows over the prompt (default on)")
    g.add_argument("--teacher", help="the engine's layer states (qwen35_forward_test --engine-states): every layer "
                   "past the first is fed the engine's residual after the layer before it, the draft the "
                   "engine's post-final-norm hidden — the strict gate's teacher-forced dump")
    g.set_defaults(func=gen_pure)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
