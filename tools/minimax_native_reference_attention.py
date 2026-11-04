"""Fixed-arithmetic Q8KV4 prefill control, independently expressed in Triton.

Inputs are contiguous, already quantized tensors. This retains the native
FP8 product chain, page BF16 boundary and FP32 accumulation order so their
effects can be separated from the unrestricted mathematical oracle. It is
not the upstream MSA GPU kernel; the default oracle remains available.
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

import minimax_native_attention_reference as oracle


@triton.jit
def _scores(q, k, out, rows, keys: tl.constexpr):
    m = tl.program_id(0)*16 + tl.arange(0, 16)
    n = tl.program_id(1)*32 + tl.arange(0, 32)
    d = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for block in range(4):
        a = tl.load(q + m[:, None]*128 + block*32 + d[None, :], m[:, None] < rows, other=0)
        b = tl.load(k + n[None, :]*128 + block*32 + d[:, None], n[None, :] < keys, other=0)
        acc = tl.dot(a.to(tl.float8e4nv, bitcast=True), b.to(tl.float8e4nv, bitcast=True), acc)
    tl.store(out + m[:, None]*keys + n[None, :], acc,
             (m[:, None] < rows) & (n[None, :] < keys))


def scores(q, k):
    shape = q.shape
    q, k = q.contiguous().view(torch.uint8), k.contiguous().view(torch.uint8)
    out = torch.empty((shape[0]*shape[1], len(k)), device=q.device, dtype=torch.float32)
    _scores[(triton.cdiv(len(out), 16), triton.cdiv(len(k), 32))](
        q, k, out, len(out), len(k), num_warps=4, enable_fp_fusion=False)
    return out.reshape(shape[0], shape[1], len(k))


@triton.jit
def _pages(q, k, v, selected, positions, partial, lse, keys: tl.constexpr,
           heads: tl.constexpr, kv_heads: tl.constexpr, topk: tl.constexpr):
    row, page, kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hpk: tl.constexpr = heads // kv_heads
    h = tl.arange(0, 16)
    t = tl.arange(0, 128)
    d = tl.arange(0, 32)
    block = tl.load(selected + (row*kv_heads+kv)*topk+page)
    pos = tl.load(positions+row)
    token = block*128+t
    visible = (block >= 0) & (token < keys) & (token <= pos)
    acc = tl.zeros((16, 128), tl.float32)
    for step in range(4):
        a = tl.load(q + (row*heads+kv*hpk+h[:, None])*128 + step*32+d[None, :],
                    h[:, None] < hpk, other=0)
        b = tl.load(k + (token[None, :]*kv_heads+kv)*128 + step*32+d[:, None],
                    visible[None, :], other=0)
        acc = tl.dot(a.to(tl.float8e4nv, bitcast=True), b.to(tl.float8e4nv, bitcast=True), acc)
    value = tl.where(visible[None, :], acc * 0.08838834764831845, -float('inf'))
    maximum = tl.max(value, axis=1)
    valid = maximum > -float('inf')
    probability = libdevice.exp(value - tl.where(valid, maximum, 0)[:, None])
    # Four sequential contributions per lane, then the five warp XOR sums.
    total = tl.zeros((16, 32), tl.float32)
    for part in tl.static_range(4):
        indices = tl.broadcast_to((d+part*32)[None, :], (16, 32))
        total = total + tl.gather(probability, indices, axis=1)
    for bit in tl.static_range(5):
        indices = tl.broadcast_to((d ^ (16 >> bit))[None, :], (16, 32))
        total = total + tl.gather(total, indices, axis=1)
    denominator = tl.sum(tl.where(d[None, :] == 0, total, 0), axis=1)
    p8 = (probability*448).to(tl.float8e4nv)
    result = tl.zeros((16, 128), tl.float32)
    for step in range(4):
        indices = tl.broadcast_to((d+step*32)[None, :], (16, 32))
        a = tl.gather(p8, indices, axis=1)
        key = block*128+step*32+d
        visible_v = (block >= 0) & (key < keys) & (key <= pos)
        b = tl.load(v + (key[:, None]*kv_heads+kv)*128 + t[None, :],
                    visible_v[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        result = tl.dot(a, b, result)
    result = tl.div_rn(result, tl.where(valid, denominator*448, 1)[:, None]).to(tl.bfloat16)
    offset = ((row*topk+page)*heads+kv*hpk+h[:, None])*128+t[None, :]
    tl.store(partial+offset, result, h[:, None] < hpk)
    logsum = tl.where(valid, maximum+libdevice.log(denominator), -float('inf'))
    tl.store(lse+(row*topk+page)*heads+kv*hpk+h, logsum, h < hpk)


@triton.jit
def _combine(partial, lse, out, heads: tl.constexpr, topk: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    maximum = tl.full((), -float('inf'), tl.float32)
    for page in range(topk):
        maximum = tl.maximum(maximum, tl.load(lse+(row*topk+page)*heads+head))
    denominator = tl.full((), 0, tl.float32)
    result = tl.zeros((128,), tl.float32)
    for page in range(topk):
        i = (row*topk+page)*heads+head
        weight = libdevice.exp(tl.load(lse+i)-maximum)
        value = tl.load(partial+i*128+d).to(tl.float32)
        result = tl.fma(value, weight, result)
        denominator = denominator+weight
    tl.store(out+(row*heads+head)*128+d, tl.div_rn(result, denominator))


def attention(q, k, v, selected, positions):
    if (q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype or v.dtype != q.dtype
            or q.shape[-1] != 128 or k.shape != v.shape or k.shape[-1] != 128
            or q.shape[1] % k.shape[1] or q.shape[1] // k.shape[1] > 16
            or selected.shape[:2] != (len(q), k.shape[1]) or positions.shape != (len(q),)
            or not 1 <= selected.shape[-1] <= 16):
        raise ValueError('native attention control geometry mismatch')
    rows, heads, kv_heads, topk = len(q), q.shape[1], k.shape[1], selected.shape[-1]
    partial = torch.empty((rows, topk, heads, 128), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((rows, topk, heads), device=q.device)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    _pages[(rows, topk, kv_heads)](q.contiguous().view(torch.uint8), k.contiguous().view(torch.uint8),
                                 v.contiguous().view(torch.uint8), selected.contiguous(), positions,
                                 partial, lse, len(k), heads, kv_heads, topk,
                                 num_warps=4, enable_fp_fusion=False)
    _combine[(rows, heads)](partial, lse, out, heads, topk, num_warps=4, enable_fp_fusion=False)
    return out


def self_test():
    torch.manual_seed(1908)
    torch.backends.cuda.matmul.allow_tf32 = False
    for rows, keys, kv_heads, hpk in ((1, 1, 1, 16), (17, 129, 2, 2), (17, 270, 2, 16), (33, 2057, 1, 16)):
        q = oracle.queries(torch.randn((rows, kv_heads*hpk, 128), device='cuda', dtype=torch.bfloat16))
        iq = oracle.queries(torch.randn((rows, kv_heads, 128), device='cuda', dtype=torch.bfloat16))
        def cache(heads):
            return oracle.unpack_cache(*oracle.pack_cache(torch.randn((keys, heads, 128), device='cuda', dtype=torch.bfloat16)))
        k, v, ik = cache(kv_heads), cache(kv_heads), cache(1)
        positions = torch.linspace(0, keys-1, rows, device='cuda').long()
        selected = oracle.select_blocks(iq, ik, positions, score_fn=scores)
        mathematical = oracle.select_blocks(iq, ik, positions)
        if not torch.equal(selected, mathematical):
            raise AssertionError('index control changes selections')
        result = attention(q, k, v, selected, positions)
        expected = oracle.attention(q, k, v, selected, positions)
        rms = ((result.double()-expected.double()).norm()/expected.double().norm()).item()
        if not math.isfinite(rms) or rms > .004:
            raise AssertionError(f'attention control exceeds the existing RMS gate: {rms}')
        for chunk in (1, 8, 17):
            split = torch.cat([attention(q[i:i+chunk], k, v, selected[i:i+chunk], positions[i:i+chunk])
                               for i in range(0, rows, chunk)])
            if not torch.equal(result, split):
                raise AssertionError('attention control changes with row geometry')
        print(f'{rows} rows, {keys} keys, {kv_heads} KV heads, {hpk} queries/KV: RMS {rms:.6g}; selections and row geometry exact', flush=True)


if __name__ == '__main__':
    self_test()
