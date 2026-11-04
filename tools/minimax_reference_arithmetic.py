"""Fixed-order elementwise controls for the independent MiniMax reference.

These Triton implementations isolate FP32 reduction/activation rounding from
matrix dispatch. They load no dgpp code. Their reduction geometry deliberately
matches the native baseline; FP64 controls below separately check the formulas.
Agreement with this control alone is not an end-to-end task-quality verdict.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as cuda_math


@triton.jit
def _norm(x, weight, out, width: tl.constexpr, eps: tl.constexpr,
          lanes: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, lanes)
    total = tl.full((lanes,), 0, tl.float32)
    for at in range(width // lanes):
        value = tl.load(x + row * width + at * lanes + lane).to(tl.float32)
        total = tl.fma(value, value, total)
    for offset in tl.static_range(0, 5):
        total = total + tl.gather(total, lane ^ (16 >> offset), axis=0)
    if lanes == 32:
        # The 128-wide Q/K norm has a constant reciprocal; CUDA contracts
        # its scale and epsilon. The residual norm divides by a runtime dim.
        variance = tl.fma(total, 1.0 / width, eps)
    else:
        value = tl.full((lanes,), 0, tl.float32)
        for warp in tl.static_range(lanes // 32):
            value = value + tl.gather(total, tl.full((lanes,), warp * 32, tl.int32), axis=0)
        variance = tl.div_rn(value, width) + eps
    inv = tl.rsqrt(variance)
    for at in range(width // lanes):
        col = at * lanes + lane
        value = tl.load(x + row * width + col).to(tl.float32)
        scale = 1.0 + tl.load(weight + col).to(tl.float32)
        tl.store(out + row * width + col, (value * inv) * scale)


@triton.jit
def _activation(gate, up, out, count, alpha: tl.constexpr, limit: tl.constexpr,
                block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    g = tl.minimum(tl.load(gate + i, i < count, other=0).to(tl.float32), limit)
    u = tl.minimum(tl.maximum(tl.load(up + i, i < count, other=0).to(tl.float32), -limit), limit)
    value = tl.div_rn(g, 1.0 + cuda_math.libdevice.exp(-alpha * g)) * (u + 1.0)
    tl.store(out + i, value, i < count)


@triton.jit
def _router(x, weight, scores, logits, width: tl.constexpr, experts: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.program_id(1) * 4 + tl.arange(0, 4)
    lane = tl.arange(0, 32)
    acc = tl.full((4, 32), 0, tl.float32)
    for at in range(width // 32):
        col = at * 32 + lane
        a = tl.load(x + row * width + col).to(tl.float32)
        w = tl.load(weight + expert[:, None] * width + col[None, :],
                    expert[:, None] < experts, other=0)
        acc = tl.fma(a[None, :], w, acc)
    for offset in tl.static_range(0, 5):
        indices = tl.broadcast_to((lane ^ (16 >> offset))[None, :], (4, 32))
        acc = acc + tl.gather(acc, indices, axis=1)
    value = tl.sum(tl.where(lane[None, :] == 0, acc, 0), axis=1)
    tl.store(logits + row * experts + expert, value, expert < experts)
    result = tl.div_rn(1.0, 1.0 + cuda_math.libdevice.exp(-value))
    tl.store(scores + row * experts + expert, result, expert < experts)


@triton.jit
def _divide(x, out, count, divisor, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.load(x + i, i < count, other=0)
    tl.store(out + i, tl.div_rn(value, divisor), i < count)


@triton.jit
def _weighted_add(acc, part, weights, rows, count, width: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    row = i // width
    to = tl.load(rows + row, i < count, other=0) * width + i % width
    a = tl.load(acc + to, i < count, other=0)
    b = tl.load(part + i, i < count, other=0)
    w = tl.load(weights + row, i < count, other=0)
    tl.store(acc + to, tl.fma(w, b, a), i < count)


def norm(x, weight, eps):
    width = x.shape[-1]
    if width not in (128, 6144):
        raise ValueError('fixed norm control supports the checkpoint widths 128 and 6144')
    if weight.numel() != width:
        raise ValueError('fixed norm weight width differs')
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    _norm[(x.numel() // width,)](x.contiguous(), weight.contiguous(), out, width, eps,
                                32 if width == 128 else 256,
                                num_warps=4 if width == 128 else 8, enable_fp_fusion=False)
    return out


def activation(gate, up, alpha, limit):
    if gate.shape != up.shape:
        raise ValueError('activation shapes differ')
    out = torch.empty(gate.shape, device=gate.device, dtype=torch.bfloat16)
    _activation[(triton.cdiv(gate.numel(), 256),)](
        gate.contiguous(), up.contiguous(), out, gate.numel(), alpha, limit, 256,
        enable_fp_fusion=False)
    return out


def router(x, weight):
    if weight.dtype != torch.float32 or x.shape[1] != weight.shape[1] or x.shape[1] % 32:
        raise ValueError('router requires F32 weights and a matching 32-aligned width')
    out = torch.empty((len(x), len(weight)), device=x.device, dtype=torch.float32)
    logits = torch.empty_like(out)
    _router[(len(x), triton.cdiv(len(weight), 4))](
        x.contiguous(), weight.contiguous(), out, logits, x.shape[1], len(weight),
        num_warps=4, enable_fp_fusion=False)
    return out, logits


def divide(x, divisor):
    out = torch.empty_like(x)
    _divide[(triton.cdiv(x.numel(), 256),)](x.contiguous(), out, x.numel(), divisor, 256,
                                          enable_fp_fusion=False)
    return out


def weighted_add(acc, part, weights, rows):
    """Scatter one expert's unique routed rows with a single FP32 FMA."""
    _weighted_add[(triton.cdiv(part.numel(), 256),)](
        acc, part.contiguous(), weights.contiguous(), rows.contiguous(), part.numel(),
        part.shape[1], 256, enable_fp_fusion=False)


def self_test():
    torch.manual_seed(1741)
    for width in (128, 6144):
        a = torch.randn((35, width), device='cuda', dtype=torch.bfloat16)
        b = torch.randn_like(a)
        weight = torch.randn((width,), device='cuda', dtype=torch.bfloat16)
        for x in (a, a.float() + b.float()):
            y = norm(x, weight, 1e-6)
            expected = (x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
                        * (1 + weight.double()))
            # One output ULP includes both BF16 rounding and the FP32 reduction
            # error. Close-to-zero weights stay exactly zero.
            rounded = expected.to(torch.bfloat16)
            ulp = torch.nextafter(rounded.abs(), torch.full_like(rounded, float('inf'))).double() - rounded.abs().double()
            assert torch.all((y.double() - expected).abs() <= ulp), (width, 'FP64 norm bound')
            for rows in (1, 8, 17):
                assert torch.equal(y, torch.cat([norm(p, weight, 1e-6) for p in x.split(rows)]))
    # Stay in the FP32 exp domain for the FP64 rounding bound. The explicit
    # saturated case below covers exp overflow in the published FP32 recipe.
    g = torch.linspace(-40, 40, 65536, device='cuda').to(torch.bfloat16)
    u = torch.randn_like(g) * 10
    actual = activation(g, u, 1.702, 7)
    gg, uu = g.double().clamp(max=7), u.double().clamp(-7, 7)
    expected = gg * torch.sigmoid(torch.tensor(1.702, dtype=torch.float32).item() * gg) * (uu + 1)
    rounded = expected.to(torch.bfloat16)
    ulp = torch.nextafter(rounded.abs(), torch.full_like(rounded, float('inf'))).double() - rounded.abs().double()
    assert torch.all((actual.double() - expected).abs() <= ulp), 'FP64 activation bound'
    extreme = torch.tensor([-100, -10000], device='cuda', dtype=torch.bfloat16)
    assert torch.equal(activation(extreme, torch.ones_like(extreme), 1.702, 7), torch.zeros_like(extreme))
    x = torch.randn((35, 6144), device='cuda', dtype=torch.bfloat16)
    w = torch.randn((19, 6144), device='cuda', dtype=torch.float32) / 80
    scores, logits = router(x, w)
    expected = x.double() @ w.double().T
    unit = torch.finfo(torch.float32).eps / 2
    gamma = (6144 // 32 + 5) * unit / (1 - (6144 // 32 + 5) * unit)
    bound = gamma * (x.double().abs() @ w.double().abs().T)
    assert torch.all((logits.double() - expected).abs() <= bound), 'FP64 router forward-error bound'
    assert torch.all((scores.double() - logits.double().sigmoid()).abs() <= 2e-7), 'sigmoid bound'
    for rows in (1, 8, 17):
        parts = [router(part, w) for part in x.split(rows)]
        assert torch.equal(scores, torch.cat([p[0] for p in parts]))
        assert torch.equal(logits, torch.cat([p[1] for p in parts]))
    a = torch.randn((11, 128), device='cuda')
    b = torch.randn((7, 128), device='cuda')
    w = torch.randn((7,), device='cuda')
    rows = torch.tensor([8, 2, 4, 0, 9, 1, 6], device='cuda')
    expected = a.clone()
    expected[rows] = (a[rows].double() + w[:, None].double() * b.double()).float()
    weighted_add(a, b, w, rows)
    assert torch.equal(a, expected), 'weighted FMA differs from FP64 rounding or touched another row'
    for scale in (1.234567, 0.005313, 19.3333):
        scale = torch.tensor(scale, dtype=torch.float32).item()
        assert torch.equal(divide(a, scale), (a.double() / scale).float()), 'FP32 global division'
    print('Fixed elementwise controls: FP64 bounds and exact row geometry passed.', flush=True)


if __name__ == '__main__':
    self_test()
