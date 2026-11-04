"""Independent fixed-order native products for MiniMax precision diagnostics.

Triton lowers dot_scaled from contiguous published payloads/scales. No DGPP
code or layouts are imported. This is an arithmetic-order control alongside
the decoded-operand cuBLAS reference, not the upstream serving GEMM. FP64
bounds and row-geometry checks validate the control independently.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _product(a, sa, w, sw, out, rows, columns: tl.constexpr, inner: tl.constexpr,
             fp4: tl.constexpr):
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    n = tl.program_id(1) * 32 + tl.arange(0, 32)
    # One native MMA instruction per K step: 32 E4M3 or 64 E2M1 values.
    k = tl.arange(0, 32)
    groups: tl.constexpr = 4 if fp4 else 1
    s = tl.arange(0, groups)
    packed_inner: tl.constexpr = inner // 2 if fp4 else inner
    scale_inner: tl.constexpr = inner // 16 if fp4 else inner // 32
    acc = tl.zeros((16, 32), tl.float32)
    for step in range(packed_inner // 32):
        av = tl.load(a + m[:, None] * packed_inner + step * 32 + k[None, :],
                     m[:, None] < rows, other=0)
        wv = tl.load(w + n[None, :] * packed_inner + step * 32 + k[:, None],
                     n[None, :] < columns, other=0)
        ass = tl.load(sa + m[:, None] * scale_inner + step * groups + s[None, :],
                      m[:, None] < rows, other=0)
        wss = tl.load(sw + n[:, None] * scale_inner + step * groups + s[None, :],
                      n[:, None] < columns, other=0)
        if fp4:
            acc = tl.dot_scaled(av, ass.to(tl.float8e4nv, bitcast=True), 'e2m1',
                                wv, wss.to(tl.float8e4nv, bitcast=True), 'e2m1', acc)
        else:
            acc = tl.dot_scaled(av, ass, 'e4m3', wv, wss, 'e4m3', acc)
    tl.store(out + m[:, None] * columns + n[None, :], acc,
             (m[:, None] < rows) & (n[None, :] < columns))


def product(a, sa, w, sw, mode):
    if mode not in ('mxfp8', 'nvfp4') or any(t.dtype != torch.uint8 for t in (a, sa, w, sw)):
        raise ValueError('native product requires uint8 payloads/scales and a supported mode')
    fp4 = mode == 'nvfp4'
    inner = a.shape[1] * (2 if fp4 else 1)
    group = 16 if fp4 else 32
    if (any(t.ndim != 2 or not t.is_contiguous() for t in (a, sa, w, sw))
            or w.shape[1] != a.shape[1] or inner % (64 if fp4 else 32)
            or sa.shape != (len(a), inner // group) or sw.shape != (len(w), inner // group)):
        raise ValueError('native product shape/stride mismatch')
    out = torch.empty((len(a), len(w)), dtype=torch.float32, device=a.device)
    _product[(triton.cdiv(len(a), 16), triton.cdiv(len(w), 32))](
        a, sa, w, sw, out, len(a), len(w), inner, fp4,
        num_warps=4, enable_fp_fusion=False)
    return out


def self_test():
    torch.manual_seed(17029)
    from minimax_reference_dump import FP4
    lut = torch.tensor(FP4, device='cuda', dtype=torch.float64)
    def operand(rows, inner, fp4):
        if not fp4:
            values = torch.randn((rows, inner), device='cuda').to(torch.float8_e4m3fn)
            scales = torch.randint(121, 132, (rows, inner // 32), device='cuda', dtype=torch.uint8)
            decoded = values.double() * torch.exp2(scales.double() - 127).repeat_interleave(32, -1)
            return values.view(torch.uint8), scales, decoded
        values = torch.randint(0, 256, (rows, inner // 2), device='cuda', dtype=torch.uint8)
        scales = torch.randint(1, 127, (rows, inner // 16), device='cuda', dtype=torch.uint8)
        codes = torch.stack((values & 15, values >> 4), -1).flatten(-2)
        decoded = lut[codes.long()] * scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, -1)
        return values, scales, decoded
    for mode in ('mxfp8', 'nvfp4'):
        for rows, columns, inner in ((1, 1, 128), (17, 73, 256), (33, 65, 6144)):
            a, sa, ad = operand(rows, inner, mode == 'nvfp4')
            w, sw, wd = operand(columns, inner, mode == 'nvfp4')
            y = product(a, sa, w, sw, mode)
            expected = ad @ wd.T
            unit = torch.finfo(torch.float32).eps / 2
            gamma = inner * unit / (1 - inner * unit)
            bound = gamma * (ad.abs() @ wd.abs().T)
            if not torch.all((y.double() - expected).abs() <= bound):
                raise AssertionError(f'{mode}: FP64 accumulation bound')
            for chunk in (1, 8, 17):
                split = torch.cat([product(a[i:i+chunk], sa[i:i+chunk], w, sw, mode)
                                   for i in range(0, rows, chunk)])
                if not torch.equal(y, split):
                    raise AssertionError(f'{mode}: row geometry')
            rms = ((y.double() - expected).norm() / expected.norm()).item()
            print(f'{mode} {rows}x{columns}x{inner}: FP64 relative RMS {rms:.6g}; exact row geometry', flush=True)


if __name__ == '__main__':
    self_test()
