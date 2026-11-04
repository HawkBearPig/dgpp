"""Activation-quantization controls for the independent MiniMax reference.

This isolates quantization error while retaining the reference's accumulation
order. It does not predict a native low-precision GEMM's reduction error or
certify its performance. No dgpp kernels are imported.
"""
import torch


def e2m1(x):
    """Saturating E2M1, round to nearest with ties to the even code."""
    magnitude = x.abs()
    code = torch.zeros_like(magnitude, dtype=torch.int64)
    for lower, midpoint in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        code += (magnitude > midpoint) | ((magnitude == midpoint) & bool(lower & 1))
    values = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6), device=x.device)
    return torch.copysign(values[code], x)


def quantize(x, mode):
    """Return an exact BF16 block payload and a separate FP32 row scale.

    MXFP8 uses E4M3 values and ceil-rounded E8M0 scales over 32 columns.
    NVFP4 uses E2M1 values, E4M3 scales over 16 columns, and a dynamic
    per-row global scale. Keeping that last factor outside the dot avoids
    an extra BF16 rounding of reconstructed activations.
    """
    if x.ndim != 2 or mode not in ('mxfp8', 'nvfp4'):
        raise ValueError('quantization requires a matrix and a known format')
    group = 32 if mode == 'mxfp8' else 16
    if x.shape[1] % group:
        raise ValueError('activation width must contain complete scale groups')
    groups = x.float().reshape(len(x), -1, group)
    maximum = groups.abs().amax(-1, keepdim=True)
    if mode == 'mxfp8':
        exponent = torch.ceil(torch.log2(maximum.clamp(min=1e-30) / 448)).clamp(-127, 127)
        scale = torch.exp2(exponent)
        codes = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return (codes * scale).reshape_as(x).to(torch.bfloat16), None
    row_max = maximum.amax(1, keepdim=True)
    global_scale = torch.where(row_max > 0, row_max / (6 * 448), 1.0)
    scale = (maximum / (6 * global_scale)).clamp(max=448).to(torch.float8_e4m3fn).float()
    inverse = torch.where(scale > 0, 1 / (scale * global_scale), 0)
    payload = e2m1(groups * inverse) * scale
    return payload.reshape_as(x).to(torch.bfloat16), global_scale.reshape(len(x), 1)


def self_test(device='cuda'):
    values = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6), device=device)
    for sign in (-1, 1):
        assert torch.equal(e2m1(sign * values), sign * values)
        midpoints = (values[:-1] + values[1:]) / 2
        expected = values[torch.tensor((0, 2, 2, 4, 4, 6, 6), device=device)]
        assert torch.equal(e2m1(sign * midpoints), sign * expected)
        toward = torch.full_like(midpoints, float('inf'))
        assert torch.equal(e2m1(sign * torch.nextafter(midpoints, toward)), sign * values[1:])
        assert torch.equal(e2m1(sign * torch.nextafter(midpoints, -toward)), sign * values[:-1])
    exact_fp4 = torch.cat((values, -values)).repeat(48).reshape(1, 768) * 448
    payload, scale = quantize(exact_fp4, 'nvfp4')
    assert torch.equal(payload.float(), exact_fp4) and torch.equal(scale, torch.ones_like(scale))
    exact_fp8 = torch.tensor((0, .001953125, .5, 1, 1.125, 64, 256, 448), device=device)
    exact_fp8 = torch.cat((exact_fp8, -exact_fp8)).repeat(48).reshape(1, 768)
    for exponent in (-8, 0, 8):
        inputs = exact_fp8 * (2.0 ** exponent)
        payload, scale = quantize(inputs, 'mxfp8')
        assert torch.equal(payload.float(), inputs) and scale is None
    torch.manual_seed(1741)
    rows = torch.randn((35, 768), device=device) * 3
    rows[0].zero_()
    for mode in ('mxfp8', 'nvfp4'):
        payload, scale = quantize(rows, mode)
        assert torch.isfinite(payload).all() and torch.count_nonzero(payload[0]) == 0
        reconstructed = payload.float() if scale is None else payload.float() * scale
        error = torch.linalg.vector_norm(reconstructed.double() - rows.double())
        rms = error / torch.linalg.vector_norm(rows.double())
        assert rms < (.04 if mode == 'mxfp8' else .14), (mode, rms)
        for width in (1, 8, 17):
            parts = [quantize(part, mode) for part in rows.split(width)]
            assert torch.equal(payload, torch.cat([p for p, _ in parts]))
            if scale is not None:
                assert torch.equal(scale, torch.cat([s for _, s in parts]))
    print(f'Activation quantization controls passed ({device})', flush=True)


if __name__ == '__main__':
    self_test()
