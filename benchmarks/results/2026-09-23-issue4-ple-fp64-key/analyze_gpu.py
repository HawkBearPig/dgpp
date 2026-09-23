#!/usr/bin/env python3
"""Require bitwise FP64-oracle partials before testing retrieval causality."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-09-23-issue4-ple-projection'
sys.path.insert(0, str(SOURCE))
from replay import bf, rb, norm, gate, statistics, sha

inputs = json.loads((SOURCE / 'gpu-input.json').read_text())
positions = inputs['positions']
rows = inputs['native_row_indices']
weights = dict(np.load(SOURCE / 'raw/weights.npz', allow_pickle=False))
checks = {}
hashes = {}
actual = {}
for mode, height in [('bf16', 2048), ('fp64', 2048), ('fp64-tail', 1024)]:
    fields = {}
    for name, width in [('key_proj0',10240), ('key_proj1',10240), ('key_proj_folded',10240),
                        ('value_proj0',2560), ('value_proj1',2560), ('value_proj_folded',2560),
                        ('key_norm',10240), ('query_norm',10240), ('gated',10240), ('conv_norm',10240)]:
        path = ROOT / f'raw/campaign/gpu-{mode}/{name}.bin'
        bits = np.fromfile(path, dtype='<u2').reshape(height, width)
        assert np.isfinite(bf(bits)).all()
        fields[name] = bits
        hashes[mode+'/'+name] = sha(path)
    actual[mode] = fields

# New build with the diagnostic off must reproduce the saved native replay.
for name, bits in actual['bf16'].items():
    original = np.fromfile(SOURCE / f'raw/gpu-campaign/bf16/{name}.bin', dtype='<u2').reshape(bits.shape)
    assert np.array_equal(bits, original), name

reference = {}
for rank in (0, 1):
    embedding = bf(np.fromfile(SOURCE / f'raw/gpu-input/embedding{rank}.bin', dtype='<u2').reshape(2048,1280)[rows])
    partial = rb(embedding.astype(np.float64) @ weights[f'key_proj{rank}'].astype(np.float64).T)
    reference[f'key_proj{rank}'] = partial
reference['key_proj_folded'] = rb(reference['key_proj0'] + reference['key_proj1'])
for mode, height in [('fp64', 2048), ('fp64-tail', 1024)]:
    selection = np.array(rows) < height
    selected_rows = np.array(rows)[selection]
    selected_positions = np.array(positions)[selection].tolist()
    for name, reference_values in reference.items():
        got = bf(actual[mode][name][selected_rows])
        expected = reference_values[selection]
        assert np.array_equal(got.view('u4'), expected.view('u4')), (mode, name)
        checks[mode+'/'+name] = statistics(expected, got, selected_positions)
    # Every unpopulated input row must remain zero after the projection.
    unpopulated = np.ones(height, dtype=bool)
    unpopulated[selected_rows] = False
    for name in reference:
        assert not bf(actual[mode][name][unpopulated]).any(), (mode, name, 'padding')

for name in ('value_proj0', 'value_proj1', 'value_proj_folded', 'query_norm'):
    assert np.array_equal(actual['bf16'][name], actual['fp64'][name]), name

base = {name: bf(bits[rows]) for name, bits in actual['bf16'].items()}
accurate = {name: bf(bits[rows]) for name, bits in actual['fp64'].items()}
for name in ('key_proj_folded', 'key_norm', 'gated', 'conv_norm'):
    checks['change/'+name] = statistics(base[name], accurate[name], positions)

# Retain the native normalization and gate: the intervention is key GEMM only.
isolated_gate, scalars = gate(accurate['key_norm'].reshape(-1,4,2560),
                            accurate['query_norm'].reshape(-1,4,2560), accurate['value_proj_folded'])
checks['gate_isolated'] = statistics(isolated_gate, accurate['gated'], positions)
key_oracle = norm(reference['key_proj_folded'], weights['norm_key'], 1e-6)
hybrid, hybrid_scalars = gate(key_oracle, base['query_norm'].reshape(-1,4,2560), base['value_proj_folded'])
checks['gate_vs_key_only_oracle'] = statistics(hybrid, accurate['gated'], positions)
worst = positions.index(136732)
assert scalars[worst, 0] == 0.0306396484375
report = dict(passed=True, native_baseline_all_fields_bitwise=True,
              key_partials_and_fold_match_fp64_oracle_bitwise=True,
              checked_shapes=[dict(rows=2048, outputs=10240, inputs=1280), dict(rows=1024, outputs=10240, inputs=1280)],
              value_and_query_unchanged=True, checks=checks,
              worst_position_136732_scalars=scalars[worst].tolist(),
              output_sha256=hashes, script_sha256=sha(Path(__file__)),
              scope='Frozen original outliers. Does not establish retrieval causality.')
(ROOT/'gpu-validation.json').write_text(json.dumps(report, indent=2)+'\n')
print('Frozen GPU preflight passed; original-request comparison may proceed.', flush=True)
