#!/usr/bin/env python3
"""Locate the first captured difference between NVFP4 and FP8 reference runs."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_requests import ordering, stats


def compare(first, second, rank):
    suffix = Path(f'rank{rank}/request1/position0')
    first, second = first / suffix, second / suffix
    assert json.loads((first / 'input.json').read_text()) == json.loads((second / 'input.json').read_text())
    fields = sorted(list(first.glob('layer-1*.json')) + list(first.glob('layer0_*.json')), key=ordering)
    assert len(fields) == 18
    results = []
    for left in fields:
        right = second / left.name
        a, b = json.loads(left.read_text()), json.loads(right.read_text())
        assert all(a[k] == b[k] for k in ('dtype', 'saved_shape', 'original_shape'))
        for path, metadata in ((left, a), (right, b)):
            data = path.with_suffix('.bin').read_bytes()
            assert len(data) == metadata['bytes']
            assert hashlib.sha256(data).hexdigest() == metadata['sha256']
        row = dict(field=left.stem, equal=a['sha256'] == b['sha256'])
        if not row['equal']:
            row['difference'] = stats(left, right, a, b)
        results.append(row)
    different = [row['field'] for row in results if not row['equal']]
    return dict(rank=rank, compared_fields=len(results), first_difference=different[0] if different else None,
                only_mlp_output_differs=different == ['layer0_mlp_out'], fields=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nvfp4', type=Path)
    parser.add_argument('fp8', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = [compare(args.nvfp4, args.fp8, rank) for rank in (0, 1)]
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in row.items() if k != 'fields'} for row in result], indent=2))
