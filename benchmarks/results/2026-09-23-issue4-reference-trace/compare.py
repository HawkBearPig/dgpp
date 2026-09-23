#!/usr/bin/env python3
"""Find the earliest observed repeated-reference difference before token drift."""
import argparse
import json
from pathlib import Path

import numpy as np

STAGES = ['ngram_context', 'query_start_loc', 'embedding', 'attn_in',
          'attn_injection', 'attn_out', 'mlp_in', 'residual',
          'mlp_injection', 'mlp_out', 'final_hidden']


def ordering(path):
    layer, stage = path.stem.split('_', 1)
    return int(layer[5:]), STAGES.index(stage)


def values(path, meta):
    dtype = meta['dtype']
    if dtype == 'torch.bfloat16':
        raw = np.fromfile(path, dtype='<u2')
        return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    types = {'torch.float32': '<f4', 'torch.float64': '<f8',
             'torch.int64': '<i8', 'torch.int32': '<i4'}
    return np.fromfile(path, dtype=types[dtype]).astype(np.float64)


def compare(root, rank):
    first, second = [root / f'rank{rank}' / f'request{i}' for i in (1, 2)]
    result = {'rank': rank, 'equal_fields': 0, 'different_fields': 0,
              'first_difference': None, 'first_input_difference': None}
    positions = sorted(first.glob('position*'), key=lambda p: int(p.name[8:]))
    for left in positions:
        right = second / left.name
        if not right.exists():
            result['first_input_difference'] = {'position': int(left.name[8:]),
                                                'reason': 'missing corresponding call'}
            break
        if json.loads((left / 'input.json').read_text()) != json.loads((right / 'input.json').read_text()):
            result['first_input_difference'] = {'position': int(left.name[8:]),
                                                'reason': 'token or position drift'}
            break
        for path in sorted(left.glob('layer*.json'), key=ordering):
            other = right / path.name
            a, b = json.loads(path.read_text()), json.loads(other.read_text())
            assert a['original_shape'] == b['original_shape']
            assert a['saved_shape'] == b['saved_shape'] and a['dtype'] == b['dtype']
            if a['sha256'] == b['sha256']:
                result['equal_fields'] += 1
                continue
            result['different_fields'] += 1
            if result['first_difference'] is None:
                x, y = values(path.with_suffix('.bin'), a), values(other.with_suffix('.bin'), b)
                indices = np.flatnonzero(x != y)
                result['first_difference'] = {
                    'position': int(left.name[8:]), 'field': path.stem,
                    'shape': a['saved_shape'], 'different_elements': int(indices.size),
                    'elements': int(x.size), 'nonfinite': int((~np.isfinite(x) | ~np.isfinite(y)).sum()),
                    'relative_l2': float(np.linalg.norm(x-y) / max(np.linalg.norm(x), 1e-30)),
                    'max_absolute': float(np.max(np.abs(x-y))),
                    'first_indices': indices[:16].tolist(),
                    'first_values': x[indices[:16]].tolist(), 'second_values': y[indices[:16]].tolist(),
                }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    args = parser.parse_args()
    print(json.dumps([compare(args.captures, rank) for rank in (0, 1)], indent=2))
