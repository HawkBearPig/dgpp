#!/usr/bin/env python3
"""Locate PLE outlier sensitivity with fixed captured operands on CPU.

The FP32 tree variant models the documented native reduction order. It is a
diagnostic comparison, not an independent oracle or a CUDA replay.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from replay import bf, rb, norm, gate, statistics, sha


def tree(values):
    assert values.shape[-1] == 2560
    partial = np.zeros(values.shape[:-1] + (256,), dtype=np.float32)
    for start in range(0, 2560, 256):
        partial = partial + values[..., start:start+256]
    partial = partial.reshape(values.shape[:-1] + (8, 32))
    lanes = np.arange(32)
    for offset in (16, 8, 4, 2, 1):
        partial = partial + partial[..., lanes ^ offset]
    total = np.zeros(values.shape[:-1], dtype=np.float32)
    for warp in range(8):
        total = total + partial[..., warp, 0]
    return total


def native_norm(values, weight):
    values = values.reshape(-1, 4, 2560)
    squared = values * values
    variance = tree(squared)[..., None] / np.float32(2560)
    rstd = (1 / np.sqrt((variance + np.float32(1e-6)).astype(np.float64))).astype(np.float32)
    return rb((values * rstd) * (np.float32(1) + weight.reshape(4, 2560)))


def native_gate(key, query, value):
    dot = rb(tree(rb(key * query)))
    scaled = rb(dot / np.float32(np.sqrt(2560.0)))
    rooted = rb(np.sqrt(np.maximum(np.abs(scaled), np.float32(1e-6)))) * np.sign(scaled)
    scalar = rb(1 / (1 + np.exp(-rooted.astype(np.float64))))
    return rb(scalar[..., None] * value[:, None, :]), scalar


def main(args):
    export = json.loads((args.rows / 'export.json').read_text())
    positions = export['selection']['positions']
    arrays = {}
    for entry in export['files']:
        rank, _, name = entry['source'].split('/')
        path = args.rows / rank / name
        assert sha(path) == entry['sample_sha256']
        arrays[(rank, name)] = bf(np.fromfile(path, dtype='<u2').reshape(len(positions), entry['width']))
    bundle = dict(np.load(args.weights, allow_pickle=False))
    assert sha(args.weights) == json.loads(args.weights.with_suffix('.json').read_text())['bundle_sha256']
    actual = arrays[('rank0', 'ple_gv.bin')].reshape(-1, 4, 2560)
    residual = arrays[('rank0', 'ple_residual_before.bin')]
    variants, projected, gate_values = [], {}, {}
    for precision in ('fp64', 'fp32'):
        dtype = np.float64 if precision == 'fp64' else np.float32
        projected[precision] = {}
        for name in ('key_proj', 'value_proj'):
            parts = [rb(arrays[(f'rank{rank}', 'ple_embedding.bin')].astype(dtype)
                        @ bundle[name+str(rank)].astype(dtype).T) for rank in (0, 1)]
            projected[precision][name] = rb(parts[0] + parts[1])
    for precision, norm_mode, gate_mode in itertools.product(('fp64', 'fp32'), ('fp64', 'tree_fp32'), ('fp64', 'tree_fp32')):
        normalize = (lambda v, w: norm(v, w, 1e-6)) if norm_mode == 'fp64' else native_norm
        calculate = gate if gate_mode == 'fp64' else native_gate
        key = normalize(projected[precision]['key_proj'], bundle['norm_key'])
        query = normalize(residual, bundle['norm_query'])
        result, scalars = calculate(key, query, projected[precision]['value_proj'])
        name = f'{precision}/{norm_mode}/{gate_mode}'
        gate_values[name] = scalars
        variants.append(dict(projection=precision, norm=norm_mode, gate=gate_mode,
                             statistics=statistics(result, actual, positions)))
    # Infer nearby BF16 gate scalars from all 2560 captured products per branch,
    # retaining an exact-match count rather than asserting an inferred value.
    inferred = []
    default = gate_values['fp64/fp64/fp64']
    value = projected['fp64']['value_proj']
    for row, position in enumerate(positions):
        branches = []
        for branch in range(4):
            code = int(default[row, branch].view('u4') >> 16)
            candidates = np.arange(max(0, code-16), min(0x3f80, code+16)+1, dtype=np.uint32)
            scalars = bf(candidates)
            matches = np.count_nonzero(rb(scalars[:, None] * value[row]) == actual[row, branch], axis=1)
            best = int(np.argmax(matches))
            branches.append(dict(branch=branch, oracle_gate=float(default[row, branch]),
                                 inferred_gate=float(scalars[best]), bf16_code_delta=int(candidates[best])-code,
                                 matching_elements=int(matches[best]), total_elements=2560,
                                 matching_candidates=int(np.count_nonzero(matches == matches[best]))))
        inferred.append(dict(position=position, branches=branches))
    report = dict(positions=positions, variants=variants, inferred_gates=inferred,
                  script_sha256=sha(Path(__file__)), source_sha256=sha(args.rows / 'export.json'),
                  scope='Fixed-input CPU rounding sensitivity and gate inference. A tree approximation is not proof of actual CUDA intermediate values or a retrieval fix.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for variant in variants:
        print(variant['projection'],variant['norm'],variant['gate'],variant['statistics']['relative_l2'],flush=True)
    print('Worst original row:', next(row for row in inferred if row['position']==136732), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
