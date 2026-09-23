#!/usr/bin/env python3
"""Place frozen outlier operands at their native row indices; pad other rows."""
import hashlib
import json
from pathlib import Path

import numpy as np

from replay import sha


root = Path(__file__).resolve().parent
source = root / 'raw/outliers'
target = root / 'raw/gpu-input'
target.mkdir(exist_ok=False)
export = json.loads((source / 'export.json').read_text())
positions = export['selection']['positions']
rows = np.array(positions) % 2048
assert len(rows) == len(set(rows.tolist()))
weights = dict(np.load(root / 'raw/weights.npz', allow_pickle=False))
for name, value in weights.items():
    (target / (name + '.bin')).write_bytes((value.view('u4') >> 16).astype('<u2').tobytes())
for rank in (0, 1):
    bits = np.fromfile(source / f'rank{rank}/ple_embedding.bin', dtype='<u2').reshape(len(rows), 1280)
    padded = np.zeros((2048, 1280), dtype='<u2')
    padded[rows] = bits
    (target / f'embedding{rank}.bin').write_bytes(padded.tobytes())
bits = np.fromfile(source / 'rank0/ple_residual_before.bin', dtype='<u2').reshape(len(rows), 10240)
padded = np.zeros((2048, 10240), dtype='<u2')
padded[rows] = bits
(target / 'residual.bin').write_bytes(padded.tobytes())
report = dict(positions=positions, native_row_indices=rows.tolist(), rows=2048,
              input_sha256={p.name: sha(p) for p in target.iterdir()},
              source_export_sha256=sha(source / 'export.json'), weights_sha256=sha(root / 'raw/weights.npz'),
              probe_sha256=sha(root / 'raw/gpu_probe'),
              control='Preserve M=2048, K/N dimensions, input stride, native row offsets, and 64 MiB workspace. Other rows are zero. Baseline must reproduce the captured outlier rows before interpreting variants.')
(root / 'gpu-input.json').write_text(json.dumps(report, indent=2)+'\n')
print('Prepared', len(rows), 'real rows at native offsets')
