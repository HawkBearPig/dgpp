#!/usr/bin/env python3
"""Independent stage checks for MiniMax's dynamic NVFP4 expert path.

With --upstream, also execute the hash-pinned FlashInfer 0.7.2 CuTe producer
and require identical payloads, scales and per-token globals. It must be
installed in the Python environment; DGPP does not depend on it at runtime.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch


def save(path, x):
    if x.dtype == torch.bfloat16:
        x = x.view(torch.uint16)
    elif x.dtype == torch.float8_e4m3fn:
        x = x.view(torch.uint8)
    x.contiguous().numpy().tofile(path)


def load(path, dtype, shape):
    return torch.from_numpy(np.fromfile(path, dtype=dtype).reshape(shape))


def generate(root):
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(129730)
    geometries = ((1, 128, 128), (17, 256, 128), (65, 128, 256), (3, 6144, 768),
                  (510, 128, 0), (5, 32768, 0))
    for i, (rows, hidden, inter) in enumerate(geometries):
        folder = root / f'case{i}'
        folder.mkdir()
        stride = hidden + 8
        x = torch.full((rows, stride), float('nan'), dtype=torch.bfloat16)
        if i == 4:
            bits = torch.arange(65536).to(torch.int32)
            bits = bits[(bits & 0x7f80) != 0x7f80].to(torch.uint16)
            x[:, :hidden] = bits.view(torch.bfloat16).reshape(rows, hidden)
        else:
            x[:, :hidden] = (torch.randn(rows, hidden) * .3).bfloat16()
            x[0, :hidden].zero_()
            x[0, 1:hidden:2] = -0.0
            if rows > 1:
                x[1, :hidden] *= 100
        save(folder / 'input.bf16', x)
        (folder / 'params.txt').write_text(f'{rows} {hidden} {inter} {stride}\n')
        (folder / 'shape.json').write_text(json.dumps(dict(rows=rows, hidden=hidden, inter=inter, stride=stride)) + '\n')
        if not inter:
            continue
        for name, n, k in (('gate', inter, hidden), ('up', inter, hidden), ('down', hidden, inter)):
            save(folder / f'{name}.codes', torch.randint(0, 256, (n, k // 2), dtype=torch.uint8))
            save(folder / f'{name}.scales', (torch.rand(n, k // 16) * .4 + .1).to(torch.float8_e4m3fn))
        save(folder / 'weight_globals.f32', torch.tensor([5., 6., 7.]))


def unpack(codes, scales):
    codes = torch.stack((codes & 15, codes >> 4), -1).flatten(-2)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
    return lut[codes.long()] * scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, -1)


def upstream_producer():
    import flashinfer
    root = Path(flashinfer.__file__).parent
    expected = {
        'quantization/kernels/nvfp4_quantize.py': '88550172a3923e6e3417d151b22b0a97d5581bc9abc378d705d737058b4a6304',
        'quantization/quantization_cute_dsl_utils.py': 'c4e2a0d938ce01953610a62f02a783aff82e877854cfab101dc79000c39f63c6',
        'cute_dsl/fp4_common.py': '6cda2e2fa955ca608844209e5792d5a2136f6a2e6e0a8d3fb7cc285ebc01531d',
    }
    for path, digest in expected.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'FlashInfer source does not match pinned 0.7.2: {path}')
    from flashinfer.quantization.kernels.nvfp4_quantize import nvfp4_quantize_per_token_cute_dsl

    def quantize(x):
        # The upstream entry requires contiguous rows; stride handling is
        # checked separately by the CUDA harness's padded input.
        result = nvfp4_quantize_per_token_cute_dsl(
            x.contiguous().cuda(), torch.tensor([1 / (6 * 448)], device='cuda'),
            sf_layout=2, enable_pdl=False, nvfp4_4over6=None)
        torch.cuda.synchronize()
        return tuple(t.cpu() for t in result)
    return quantize


def check_quant(folder, prefix, x, producer):
    rows, k = x.shape
    codes = load(folder / f'{prefix}codes.u8', np.uint8, (rows, k // 2))
    scales = load(folder / f'{prefix}scales.u8', np.uint8, (rows, k // 16))
    globals_ = load(folder / f'{prefix}globals.f32', np.float32, (rows,))
    reference_global = x.float().abs().amax(-1) * torch.tensor(1 / (6 * 448), dtype=torch.float32)
    assert torch.equal(globals_.view(torch.int32), reference_global.view(torch.int32)), 'row scale formula'
    if producer:
        expected = producer(x)
        for name, got, want in zip(('codes', 'scales', 'globals'), (codes, scales, globals_), expected):
            got_bytes, want_bytes = got.view(torch.uint8).flatten(), want.view(torch.uint8).flatten()
            assert torch.equal(got_bytes, want_bytes), (
                folder.name, prefix, name, 'upstream byte mismatch', int((got_bytes != want_bytes).sum()))
    decoded = unpack(codes, scales) * globals_.double()[:, None]
    # A format-level rounding bound, independent of the GPU reciprocal
    # approximation. E2M1's largest half-interval is one scale unit.
    if torch.isfinite(decoded).all():
        unit = scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, -1) * globals_.double()[:, None]
        # Zero E4M3 groups have a separate underflow allowance.
        underflow = globals_.double()[:, None] * (3 / 1024)
        # The publisher's rcp.approx.ftz flushes subnormal row scales before
        # reciprocal, so those exhaustive BF16 edge rows do not obey an
        # ordinary nearest-format rounding bound. They remain covered by
        # the exact upstream payload/scale comparison above.
        regular = (globals_ >= torch.finfo(torch.float32).tiny) | (reference_global == 0)
        error = (decoded - x.double()).abs()
        bound = unit + underflow + x.double().abs() * 1e-6
        assert torch.all(error[regular] <= bound[regular]), 'NVFP4 rounding bound'
    return decoded, codes, scales, globals_


def product_bound(a, b, scale, actual, bf16=False):
    raw = a @ b.T
    expected = raw * scale[:, None]
    unit = torch.finfo(torch.float32).eps / 2
    gamma = a.shape[1] * unit / (1 - a.shape[1] * unit)
    bound = gamma * (a.abs() @ b.abs().T) * scale.abs()[:, None] + expected.abs() * (4 * unit)
    if bf16:
        # Half a BF16 ULP at the reference magnitude, with a conservative
        # exponent-boundary allowance. Forward summation error is separate.
        exponent = torch.floor(torch.log2(expected.abs().clamp_min(2.0**-126)))
        bound += torch.exp2(exponent - 8)
    error = (actual.double() - expected).abs()
    assert torch.all(error <= bound + 1e-10), ('matrix FP64 forward-error bound', float((error - bound).max()))
    return float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp_min(1e-30))


def check(folder, producer):
    shape = json.loads((folder / 'shape.json').read_text())
    rows, hidden, inter, stride = (shape[n] for n in ('rows', 'hidden', 'inter', 'stride'))
    x = load(folder / 'input.bf16', np.uint16, (rows, stride)).view(torch.bfloat16)[:, :hidden]
    _, codes, scales, globals_ = check_quant(folder, 'actual_', x, producer)
    result = dict(**shape)
    if inter:
        weight_globals = load(folder / 'weight_globals.f32', np.float32, (3,))
        weights = {}
        for name, n, k in (('gate', inter, hidden), ('up', inter, hidden), ('down', hidden, inter)):
            weights[name] = unpack(load(folder / f'{name}.codes', np.uint8, (n, k // 2)),
                                   load(folder / f'{name}.scales', np.uint8, (n, k // 16)))
        a = unpack(codes, scales)
        projections = {}
        for i, name in enumerate(('gate', 'up')):
            projections[name] = load(folder / f'actual_{name}.f32', np.float32, (rows, inter))
            # The GEMM epilogue multiplies the row global by the reciprocal
            # checkpoint encode global once, after FP32 accumulation.
            factor = (globals_ * weight_globals[i].reciprocal()).double()
            result[f'{name}_relative_rms'] = product_bound(a, weights[name], factor, projections[name])
        g, u = projections['gate'].clamp(max=7), projections['up'].clamp(-7, 7)
        expected_act = (g / (1 + torch.exp(-1.702 * g)) * (u + 1)).bfloat16()
        act = load(folder / 'actual_activation.bf16', np.uint16, (rows, inter)).view(torch.bfloat16)
        assert torch.equal(act.view(torch.uint16), expected_act.view(torch.uint16)), 'SwiGLU FP32-to-BF16 boundary'
        _, ac, ass, ag = check_quant(folder, 'actual_act_', act, producer)
        output = load(folder / 'actual_output.bf16', np.uint16, (rows, hidden)).view(torch.bfloat16)
        result['down_relative_rms'] = product_bound(unpack(ac, ass), weights['down'],
                                                    (ag * weight_globals[2].reciprocal()).double(), output, True)
    print(f'{folder.name}: stage checks passed; upstream byte comparison={bool(producer)}', flush=True)
    return result


def run(args):
    if not args.check_only:
        generate(args.out)
    if args.generate_only:
        return
    if not args.check_only:
        subprocess.run([str(args.app.absolute()), str(args.out.absolute())], check=True)
    producer = upstream_producer() if args.upstream else None
    results = [check(folder, producer) for folder in sorted(args.out.glob('case*'))]
    report = dict(ok=True, cases=results, upstream_byte_exact=bool(producer), torch_version=torch.__version__)
    if args.app:
        report['app_sha256'] = hashlib.sha256(args.app.read_bytes()).hexdigest()
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--upstream', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.generate_only and args.check_only:
        parser.error('choose only one partial phase')
    if (args.generate_only or args.check_only) and args.out is None:
        parser.error('--out is required for a partial phase')
    if not args.generate_only and not args.check_only and args.app is None:
        parser.error('--app is required')
    if args.out:
        run(args)
    else:
        with tempfile.TemporaryDirectory(prefix='dgpp_minimax_nvfp4_') as directory:
            args.out = Path(directory) / 'fixtures'
            run(args)


if __name__ == '__main__':
    main()
