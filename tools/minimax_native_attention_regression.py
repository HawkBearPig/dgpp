#!/usr/bin/env python3
"""Fixture and independent precision-formula checks for DGPP's Q8KV4 kernels."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

import minimax_native_attention_reference as reference


def save(path, tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    elif tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.uint8)
    tensor.contiguous().numpy().tofile(path)


def make_case(root, index, rows, requests, blocks, kv_heads, heads_per_kv):
    folder = root / f'case{index}'
    folder.mkdir()
    heads, topk = kv_heads * heads_per_kv, 16
    span = blocks * 128
    tokens = requests * span
    table = torch.randperm(requests * blocks).reshape(requests, blocks).to(torch.int32)
    pos_values = (0, 1, 31, 32, 127, 128, 129, 2047, 2048, span - 1, -1)
    pos = torch.tensor([min(pos_values[row % len(pos_values)], span - 1) for row in range(rows)])
    req = torch.tensor([row % requests if pos[row] >= 0 else -1 for row in range(rows)], dtype=torch.int32)
    inputs = {}
    for name, shape in (('q', (rows, heads, 128)), ('iq', (rows, kv_heads, 128)),
                        ('k', (tokens + 1, kv_heads, 128)), ('v', (tokens + 1, kv_heads, 128)),
                        ('ik', (tokens + 1, 1, 128))):
        inputs[name] = (torch.randn(shape) * .5).to(torch.bfloat16)
    inputs['iq'][::13].zero_()  # exact index ties; forced current-page selection
    inputs['q'][::11].zero_()   # uniform per-page probabilities
    for name in ('k', 'v', 'ik'):
        inputs[name][-1].fill_(float('nan'))  # invalid append must never read it
        if name != 'v':
            inputs[name][0].zero_()
    append_req = torch.cat((torch.arange(tokens) // span, torch.tensor([-1]))).to(torch.int32)
    append_pos = torch.cat((torch.arange(tokens) % span, torch.tensor([-1])))
    (folder / 'params.txt').write_text(f'{rows} {tokens + 1} {requests} {blocks} {heads} {kv_heads} {topk}\n')
    (folder / 'shape.json').write_text(json.dumps(dict(rows=rows, requests=requests, blocks=blocks,
                                                     heads=heads, kv_heads=kv_heads, topk=topk)) + '\n')
    for name, value in inputs.items():
        save(folder / f'{name}.bf16', value)
    for name, value, dtype in (('table', table, 'i32'), ('req', req, 'i32'),
                               ('positions', pos, 'i64'), ('append_req', append_req, 'i32'),
                               ('append_pos', append_pos, 'i64')):
        save(folder / f'{name}.{dtype}', value)


def load(path, dtype, shape):
    return torch.from_numpy(np.fromfile(path, dtype=dtype).reshape(shape))


def check_case(folder):
    shape = json.loads((folder / 'shape.json').read_text())
    rows, requests, blocks, heads, kv_heads, topk = (shape[x] for x in ('rows', 'requests', 'blocks', 'heads', 'kv_heads', 'topk'))
    span, tokens = blocks * 128, requests * blocks * 128
    req = load(folder / 'req.i32', np.int32, (rows,)).long()
    pos = load(folder / 'positions.i64', np.int64, (rows,))
    table = load(folder / 'table.i32', np.int32, (requests, blocks)).long()
    q = reference.queries(load(folder / 'q.bf16', np.uint16, (rows, heads, 128)).view(torch.bfloat16))
    iq = reference.queries(load(folder / 'iq.bf16', np.uint16, (rows, kv_heads, 128)).view(torch.bfloat16))
    for name, value in (('q', q), ('iq', iq)):
        got = load(folder / f'actual_{name}.u8', np.uint8, value.shape)
        assert torch.equal(got, value.view(torch.uint8)), f'{name}: query conversion'
    logical = torch.arange(tokens)
    physical = table[logical // span, logical % span // 128] * 128 + logical % 128
    caches = {}
    for name, h in (('k', kv_heads), ('v', kv_heads), ('ik', 1)):
        x = load(folder / f'{name}.bf16', np.uint16, (tokens + 1, h, 128)).view(torch.bfloat16)[:tokens]
        codes, scales = reference.pack_cache(x)
        expected = torch.empty((tokens, h, 72), dtype=torch.uint8)
        expected[physical, :, :64] = codes
        expected[physical, :, 64:] = scales.view(torch.uint8)
        got = load(folder / f'actual_{name}.u8', np.uint8, expected.shape)
        assert torch.equal(got, expected), f'{name}: packed cache mismatch ({int((got != expected).sum())} bytes)'
        caches[name] = reference.unpack_cache(codes, scales)
    selected = load(folder / 'actual_selected.i32', np.int32, (rows, kv_heads, topk))
    scores = load(folder / 'actual_scores.f32', np.float32, (rows, kv_heads, blocks))
    actual = load(folder / 'actual_out.bf16', np.uint16, (rows, heads, 128)).view(torch.bfloat16)
    actual_decode = load(folder / 'actual_decode.bf16', np.uint16, actual.shape).view(torch.bfloat16)
    expected = torch.zeros_like(actual)
    expected_decode = torch.zeros_like(actual)
    max_index_error = 0.0
    unit = torch.finfo(torch.float32).eps / 2
    gamma = 128 * unit / (1 - 128 * unit)
    for request in range(requests):
        at = torch.where((req == request) & (pos >= 0))[0]
        if not len(at):
            continue
        qs = q.view(torch.uint8)[at].view(torch.float8_e4m3fn)
        iqs = iq.view(torch.uint8)[at].view(torch.float8_e4m3fn)
        k, v, ik = (caches[name][request * span:(request + 1) * span] for name in ('k', 'v', 'ik'))
        chosen = reference.select_blocks(iqs, ik, pos[at], topk)
        assert torch.equal(selected[at], chosen), 'index selection differs from independent FP32 reference'
        dots = torch.einsum('rhd,td->rht', iqs.double(), ik[:, 0].double())
        bound = gamma * torch.einsum('rhd,td->rht', iqs.double().abs(), ik[:, 0].double().abs())
        for item, row in enumerate(at.tolist()):
            for page in range(int(pos[row]) // 128 + 1):
                begin, end = page * 128, min((page + 1) * 128, int(pos[row]) + 1)
                maximum = dots[item, :, begin:end].amax(-1)
                error = (scores[row, :, page].double() - maximum).abs()
                assert torch.all(error <= bound[item, :, begin:end].amax(-1) + 1e-10), 'FP64 index error bound'
                max_index_error = max(max_index_error, float(error.max()))
        expected[at] = reference.attention(qs, k, v, chosen, pos[at], 'prefill')
        expected_decode[at] = reference.attention(qs, k, v, chosen, pos[at], 'decode')
    assert torch.equal(selected[pos < 0], torch.full_like(selected[pos < 0], -1)), 'padding selected pages'
    assert torch.count_nonzero(actual[pos < 0]) == 0, 'padding attention output'
    assert torch.count_nonzero(actual_decode[pos < 0]) == 0, 'padding decode output'
    relative = float(torch.linalg.vector_norm(actual.double() - expected.double()) /
                     torch.linalg.vector_norm(expected.double()).clamp_min(1e-30))
    # The existing MiniMax isolated attention gate; the formula already
    # includes query/cache/probability quantization and page BF16 rounding.
    assert torch.isfinite(actual).all() and relative <= .004, ('attention formula', relative)
    decode_relative = float(torch.linalg.vector_norm(actual_decode.double() - expected_decode.double()) /
                            torch.linalg.vector_norm(expected_decode.double()).clamp_min(1e-30))
    assert torch.isfinite(actual_decode).all() and decode_relative <= .004, ('decode formula', decode_relative)
    print(f'{folder.name}: cache/query bytes exact, index selections exact, prefill relRMS={relative:.7g}, decode relRMS={decode_relative:.7g}', flush=True)
    return dict(**shape, max_index_abs_error=max_index_error, attention_relative_rms=relative,
                decode_relative_rms=decode_relative)


def run(args):
    if not args.check_only:
        args.out.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(45129)
        for i, geometry in enumerate(((1, 1, 1, 1, 16), (17, 2, 2, 1, 16),
                                       (33, 3, 17, 2, 4), (65, 3, 17, 1, 16),
                                       (8, 2, 2, 4, 16))):
            make_case(args.out, i, *geometry)
    if args.generate_only:
        return
    if not args.check_only:
        subprocess.run([str(args.app.absolute()), str(args.out.absolute())], check=True)
    results = [check_case(path) for path in sorted(args.out.glob('case*'))]
    report = dict(ok=True, cases=results, torch_version=torch.__version__)
    if args.app:
        report['app_sha256'] = hashlib.sha256(args.app.read_bytes()).hexdigest()
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='retain fixtures and results here; otherwise use temporary storage')
    parser.add_argument('--app', type=Path)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.generate_only and args.check_only:
        parser.error('choose only one partial phase')
    if not args.generate_only and not args.check_only and args.app is None:
        parser.error('--app is required to run the CUDA harness')
    if (args.generate_only or args.check_only) and args.out is None:
        parser.error('--out is required for a partial phase')
    if args.out is not None:
        run(args)
    else:
        with tempfile.TemporaryDirectory(prefix='dgpp_minimax_q8kv4_') as directory:
            args.out = Path(directory) / 'fixtures'
            run(args)


if __name__ == '__main__':
    main()
