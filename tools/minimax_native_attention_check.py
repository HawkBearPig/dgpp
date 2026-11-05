#!/usr/bin/env python3
"""Check the contiguous Q8KV4 oracle against pinned, independently authored MSA.

Runs on CPU without loading a model or initializing CUDA. Supply an MSA
checkout at c8e1013a; the three reference files are checked by SHA-256 before
loading their standalone functions. No upstream GPU package is imported.
"""

import argparse
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

import minimax_native_attention_reference as reference


SOURCES = {
    'prefill': ('tests/inference/msa_v1/attention/prefill/q8kv4/reference.py',
                '8ea5fc35b75d4a127d131a9426c4220d271c2d8099ed01fa566c85d0b1256572',
                {'dequantize_cache', 'paged_sparse_attention_reference'}),
    'decode': ('tests/inference/msa_v1/attention/decode/reference.py',
               '50c8d9988d9d50f8a6d7d8540a113bae27ce6f5f4a4a847f45dc2c7b19abba6d',
               {'sparse_decode_reference'}),
    'convert': ('tests/inference/dequant/nvfp4_to_fp8/reference.py',
                '289d810ea00f93a04125b4e2806da2bc3f99619eb0d928e02b265c617434f4a0',
                {'dequantize_reference', 'make_exhaustive_inputs'}),
}


def upstream(root, name):
    relative, expected, names = SOURCES[name]
    path = root / relative
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'{path}: expected the pinned MSA c8e1013a reference')
    tree = ast.parse(data)
    # Postpone the annotations of upstream fixture classes; retain the actual
    # reference bodies and constants unchanged, without fixture/GPU imports.
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [node for node in tree.body
              if isinstance(node, (ast.Assign, ast.AnnAssign))
              or isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'torch': torch}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def check(root):
    torch.set_num_threads(1)
    torch.manual_seed(30419)
    prefill, decode, convert = (upstream(root, name) for name in SOURCES)
    packed, scales = convert['make_exhaustive_inputs'](torch.device('cpu'))
    expected = convert['dequantize_reference'](packed, scales)
    actual = reference.unpack_cache(packed, scales)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), 'cache conversion'

    # Exactly representable cache values exercise all magnitudes, signed zero,
    # packing order and group scales without deriving the expected codes by
    # calling the quantizer under test.
    codes = torch.arange(128, dtype=torch.uint8) % 16
    payload = (codes[::2] | codes[1::2] << 4)[None, :]
    unit = torch.ones((1, 8)).to(torch.float8_e4m3fn)
    values = reference.unpack_cache(payload, unit).to(torch.bfloat16)
    repacked, rescaled = reference.pack_cache(values)
    assert torch.equal(payload, repacked), 'E2M1 packing/sign'
    assert torch.equal(unit.view(torch.uint8), rescaled.view(torch.uint8)), 'unit cache scale'
    zero, zero_scale = reference.pack_cache(torch.zeros((2, 128), dtype=torch.bfloat16))
    assert not zero.any() and not zero_scale.float().any(), 'zero cache groups'

    different_phases = 0
    for length, rows in ((1, 1), (129, 3), (270, 8), (2057, 3)):
        pages = (length + 127) // 128
        k = torch.randn((pages * 128, 4, 128), dtype=torch.bfloat16) * .5
        v = torch.randn_like(k) * .5
        q = reference.queries(torch.randn((rows, 64, 128), dtype=torch.bfloat16) * .5)
        packed_k, scale_k = reference.pack_cache(k)
        packed_v, scale_v = reference.pack_cache(v)
        k = reference.unpack_cache(packed_k, scale_k)
        v = reference.unpack_cache(packed_v, scale_v)
        positions = torch.arange(length - rows, length)
        selected = torch.full((rows, 4, 16), -1, dtype=torch.int32)
        for row, position in enumerate(positions.tolist()):
            for head in range(4):
                visible = list(range(position // 128 + 1))
                if len(visible) > 16:
                    visible.remove(head)
                selected[row, head, :len(visible)] = torch.tensor(visible)

        # Upstream sees an independently permuted physical page table. The
        # local oracle only sees contiguous logical keys and values.
        order = torch.randperm(pages)
        table = torch.argsort(order).to(torch.int32)[None, :]

        def page(x):
            raw = x.reshape(pages, 128, 4, -1).transpose(1, 2).contiguous()
            if raw.dtype == torch.float8_e4m3fn:
                return raw.view(torch.uint8)[order].view(torch.float8_e4m3fn)
            return raw[order]

        expected, _ = prefill['paged_sparse_attention_reference'](
            q, page(packed_k), page(packed_v), page(scale_k), page(scale_v), table,
            selected.transpose(0, 1).contiguous(), (rows,), (length,))
        actual_prefill = reference.attention(q, k, v, selected, positions, 'prefill')
        assert torch.equal(actual_prefill.float(), expected), ('upstream prefill', length)
        inputs = SimpleNamespace(q=q, topk_indices=selected, page_table=table,
                                 seq_lens=torch.tensor([length]), q_len_per_req=rows)
        expected = decode['sparse_decode_reference'](
            inputs, page(k), page(v), softmax_scale=128 ** -.5)
        actual_decode = reference.attention(q, k, v, selected, positions, 'decode')
        assert torch.equal(actual_decode, expected), ('upstream decode', length)
        different_phases += int((actual_prefill != actual_decode).sum())
        for chunk in (1, 3):
            for phase, actual in (('prefill', actual_prefill), ('decode', actual_decode)):
                assert torch.equal(actual, reference.attention(q, k, v, selected, positions, phase, chunk)), (
                    'reference query chunk invariance', length, phase, chunk)
        print(f'Pinned MSA prefill/decode exact: keys={length}, queries={rows}', flush=True)

    assert different_phases > 0, 'fixture must distinguish upstream phase formulas'
    print(f'Q8KV4 oracle passed; {different_phases} output elements differ between upstream phases.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--msa-source', type=Path, required=True)
    check(parser.parse_args().msa_source)


if __name__ == '__main__':
    main()
