"""Independent fixed-order FP32 MSA control, with separate FP64 formula checks.

Uses contiguous reference tensors, never dgpp's paged buffers or CUDA kernels.
The scalar-dot, 32-key online-softmax and sorted-block fold order match the
conservative native baseline to isolate rounding from mathematical defects.
This diagnostic does not replace the unrestricted PyTorch or task-quality gates.
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as cuda_math


@triton.jit
def _index(q, k, positions, out, heads: tl.constexpr, blocks: tl.constexpr,
           keys: tl.constexpr):
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    position = tl.load(positions + row)
    token = block * 128 + tl.arange(0, 128)
    dot = tl.full((128,), 0, tl.float32)
    if block * 128 <= position:
        for d in range(128):
            query = tl.load(q + (row * heads + head) * 128 + d).to(tl.float32)
            key = tl.load(k + token * 128 + d, (token <= position) & (token < keys), other=0).to(tl.float32)
            dot = tl.fma(query, key, dot)
        dot = tl.where((token <= position) & (token < keys), dot, -float('inf'))
        maximum = tl.max(dot, axis=0)
    else:
        maximum = -float('inf')
    tl.store(out + (row * heads + head) * blocks + block, maximum)


@triton.jit
def _partial(q, k, v, selected, positions, maxima, sums, values,
             heads: tl.constexpr, kv_heads: tl.constexpr, splits: tl.constexpr,
             keys: tl.constexpr, scale: tl.constexpr):
    row, split, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv = head // (heads // kv_heads)
    block = tl.load(selected + (row * kv_heads + kv) * splits + split)
    position = tl.load(positions + row)
    end = tl.minimum(tl.minimum(block * 128 + 128, position + 1), keys)
    lane = tl.arange(0, 32)
    dim = tl.arange(0, 128)
    maximum = -float('inf')
    total = 0.0
    acc = tl.full((128,), 0, tl.float32)
    if block >= 0:
        for start in range(block * 128, end, 32):
            count = tl.minimum(32, end - start)
            token = start + lane
            score = tl.full((32,), 0, tl.float32)
            for d in range(128):
                query = tl.load(q + (row * heads + head) * 128 + d).to(tl.float32)
                key = tl.load(k + (token * kv_heads + kv) * 128 + d, lane < count, other=0).to(tl.float32)
                score = tl.fma(query, key, score)
            score = tl.where(lane < count, score * scale, -float('inf'))
            next_maximum = tl.maximum(maximum, tl.max(score, axis=0))
            rescale = cuda_math.libdevice.exp(maximum - next_maximum)
            probability = tl.where(lane < count, cuda_math.libdevice.exp(score - next_maximum), 0.0)
            subtotal = probability
            for offset in tl.static_range(0, 5):
                subtotal = subtotal + tl.gather(subtotal, lane ^ (16 >> offset), axis=0)
            subtotal = tl.sum(tl.where(lane == 0, subtotal, 0.0), axis=0)
            total = tl.fma(total, rescale, subtotal)
            acc = acc * rescale
            for t in range(count):
                p = tl.sum(tl.gather(probability, tl.full((1,), t, tl.int32), axis=0), axis=0)
                value = tl.load(v + ((start + t) * kv_heads + kv) * 128 + dim).to(tl.float32)
                acc = tl.fma(p, value, acc)
            maximum = next_maximum
    at = (row * splits + split) * heads + head
    tl.store(maxima + at, maximum)
    tl.store(sums + at, total)
    tl.store(values + at * 128 + dim, acc)


@triton.jit
def _combine(maxima, sums, values, out, heads: tl.constexpr, splits: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    dim = tl.arange(0, 128)
    maximum = -float('inf')
    for split in range(splits):
        at = (row * splits + split) * heads + head
        maximum = tl.maximum(maximum, tl.load(maxima + at))
    total = 0.0
    value = tl.full((128,), 0, tl.float32)
    if maximum > -float('inf'):
        for split in range(splits):
            at = (row * splits + split) * heads + head
            weight = cuda_math.libdevice.exp(tl.load(maxima + at) - maximum)
            total = tl.fma(tl.load(sums + at), weight, total)
            value = tl.fma(tl.load(values + at * 128 + dim), weight, value)
    tl.store(out + (row * heads + head) * 128 + dim,
             tl.where(total > 0, tl.div_rn(value, total), 0.0))


def index_scores(q, k, positions):
    if q.shape[-1] != 128 or k.numel() != len(k) * 128 or len(positions) != len(q):
        raise ValueError('fixed index requires 128-wide heads')
    rows, heads = q.shape[:2]
    blocks = triton.cdiv(len(k), 128)
    result = torch.empty((rows, heads, blocks), device=q.device, dtype=torch.float32)
    _index[(rows, heads, blocks)](q.contiguous(), k.contiguous(), positions.contiguous(),
                                  result, heads, blocks, len(k), num_warps=4, enable_fp_fusion=False)
    return result


def attention(q, k, v, selected, positions, query_rows=128):
    rows, heads, dim = q.shape
    kv_heads = k.shape[1]
    if (dim != 128 or k.shape[-1] != 128 or k.shape != v.shape or heads % kv_heads
            or selected.shape[:2] != (rows, kv_heads) or len(positions) != rows
            or any(x.dtype != torch.bfloat16 for x in (q, k, v))):
        raise ValueError('fixed attention geometry mismatch')
    if query_rows < 1:
        raise ValueError('query_rows must be positive')
    splits = selected.shape[-1]
    if not 1 <= splits <= 16:
        raise ValueError('fixed attention needs one to sixteen selected blocks')
    k, v = k.contiguous(), v.contiguous()
    result = torch.empty_like(q)
    for start in range(0, rows, query_rows):
        stop = min(start + query_rows, rows)
        count = stop - start
        maximum = torch.empty((count, splits, heads), device=q.device, dtype=torch.float32)
        total = torch.empty_like(maximum)
        values = torch.empty((count, splits, heads, 128), device=q.device, dtype=torch.float32)
        _partial[(count, splits, heads)](
            q[start:stop].contiguous(), k, v, selected[start:stop].contiguous(),
            positions[start:stop].contiguous(), maximum, total, values, heads, kv_heads,
            splits, len(k), 1 / math.sqrt(128), num_warps=4, enable_fp_fusion=False)
        _combine[(count, heads)](maximum, total, values, result[start:stop], heads, splits,
                                 num_warps=4, enable_fp_fusion=False)
    return result


def select_blocks(q, k, positions, topk=16):
    scores = index_scores(q, k, positions)
    current = (positions // 128)[:, None, None].expand(len(q), q.shape[1], 1)
    scores.scatter_(2, current, torch.inf)
    selected = scores.argsort(dim=-1, descending=True, stable=True)[..., :topk].sort(-1).values
    selected = torch.where(selected * 128 <= positions[:, None, None], selected, -1)
    if selected.shape[-1] < topk:
        selected = torch.nn.functional.pad(selected, (0, topk - selected.shape[-1]), value=-1)
    return selected.to(torch.int32)


def self_test():
    torch.manual_seed(4913)
    unit = torch.finfo(torch.float32).eps / 2
    gamma = 128 * unit / (1 - 128 * unit)
    for length in (1, 33, 129, 2049):
        points = list(range(length)) if length <= 129 else [0, 31, 32, 127, 128, 129, 2047, 2048]
        pos = torch.tensor(points, device='cuda', dtype=torch.int64)
        q = torch.randn((len(pos), 4, 128), device='cuda', dtype=torch.bfloat16)
        k = torch.randn((length, 2, 128), device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        selected = torch.full((len(pos), 2, 16), -1, device='cuda', dtype=torch.int32)
        for row, position in enumerate(points):
            for kv in range(2):
                blocks = list(range(position // 128 + 1))
                if len(blocks) > 16:
                    blocks.remove(15 - kv)
                selected[row, kv, :len(blocks)] = torch.tensor(blocks, device='cuda')
        actual = attention(q, k, v, selected, pos)
        for row, position in enumerate(points):
            for head in range(4):
                kv = head // 2
                ids = [128 * block + off for block in selected[row, kv].tolist() if block >= 0
                       for off in range(128) if 128 * block + off <= position]
                kk, vv = k[ids, kv].double(), v[ids, kv].double()
                scores = (kk @ q[row, head].double()) / math.sqrt(128)
                probabilities = scores.softmax(0)
                expected = probabilities @ vv
                # BF16 output rounding plus a conditioned FP32 attention
                # allowance; near-zero sums use the weighted absolute values.
                bound = .004 * expected.abs() + .0002 * (probabilities @ vv.abs()) + 1e-8
                assert torch.all((actual[row, head].double() - expected).abs() <= bound), ('FP64 attention', length, position, head)
        for batch in (1, 8, 17):
            assert torch.equal(actual, attention(q, k, v, selected, pos, batch)), ('query geometry', length, batch)
        iq, ik = q[:, :2].contiguous(), k[:, :1].contiguous()
        scores = index_scores(iq, ik, pos)
        kk = ik[:, 0].double()
        for row, position in enumerate(points):
            dot = iq[row].double() @ kk.T
            error = gamma * (iq[row].double().abs() @ kk.abs().T)
            for block in range(scores.shape[-1]):
                start, stop = block * 128, min((block + 1) * 128, position + 1)
                if start >= stop:
                    assert torch.isneginf(scores[row, :, block]).all()
                    continue
                expected = dot[:, start:stop].max(-1).values
                bound = error[:, start:stop].max(-1).values
                assert torch.all((scores[row, :, block].double() - expected).abs() <= bound), 'FP64 index dot bound'
        chosen = select_blocks(iq, ik, pos)
        for row, position in enumerate(points):
            for head in range(2):
                blocks = chosen[row, head].tolist()
                visible = [b for b in blocks if b >= 0]
                assert position // 128 in visible and len(set(visible)) == len(visible)
                assert visible == sorted(visible) and len(visible) == min(16, position // 128 + 1)
    print('Fixed MSA control: FP64 bounds, causal/block boundaries and exact query geometry passed.', flush=True)


if __name__ == '__main__':
    self_test()
