#!/usr/bin/env python3
"""Compare captured GPU intermediates with the independent CPU replay."""
import json
from pathlib import Path

import numpy as np

from replay import bf, rb, norm, gate, statistics, sha
from probe_outliers import native_norm, native_gate


root = Path(__file__).resolve().parent
inputs = json.loads((root / 'gpu-input.json').read_text())
positions, rows = inputs['positions'], inputs['native_row_indices']
weights = dict(np.load(root / 'raw/weights.npz', allow_pickle=False))
source = root / 'raw/outliers'
residual = bf(np.fromfile(source / 'rank0/ple_residual_before.bin', dtype='<u2').reshape(len(rows), 10240))
actual = {}
hashes = {}
for mode in ('bf16', 'fp32'):
    actual[mode] = {}
    for field, width in [('key_proj0',10240),('key_proj1',10240),('key_proj_folded',10240),
                         ('value_proj0',2560),('value_proj1',2560),('value_proj_folded',2560),
                         ('key_norm',10240),('query_norm',10240),('gated',10240),('conv_norm',10240)]:
        path = root / f'raw/gpu-campaign/{mode}/{field}.bin'
        actual[mode][field] = bf(np.fromfile(path,dtype='<u2').reshape(2048,width)[rows])
        hashes[mode+'/'+field] = sha(path)
    for field, captured in (('gated','ple_gv'),('conv_norm','ple_un')):
        prior = bf(np.fromfile(source/f'rank0/{captured}.bin',dtype='<u2').reshape(len(rows),10240))
        assert np.array_equal(actual[mode][field],prior),(mode,field)

gpu = actual['bf16']
checks = {}
projected = {}
for name in ('key_proj','value_proj'):
    parts = []
    for rank in (0,1):
        embedding = bf(np.fromfile(source/f'rank{rank}/ple_embedding.bin',dtype='<u2').reshape(len(rows),1280))
        partial = rb(embedding.astype(np.float64) @ weights[name+str(rank)].astype(np.float64).T)
        parts.append(partial)
        checks[name+str(rank)] = statistics(partial,gpu[name+str(rank)],positions)
    projected[name] = rb(parts[0]+parts[1])
    checks[name+'_folded'] = statistics(projected[name],gpu[name+'_folded'],positions)
key = norm(projected['key_proj'],weights['norm_key'],1e-6)
query = norm(residual,weights['norm_query'],1e-6)
checks['key_norm_chain'] = statistics(key,gpu['key_norm'],positions)
checks['key_norm_isolated'] = statistics(norm(gpu['key_proj_folded'],weights['norm_key'],1e-6),gpu['key_norm'],positions)
checks['key_norm_native_tree'] = statistics(native_norm(gpu['key_proj_folded'],weights['norm_key']),gpu['key_norm'],positions)
checks['query_norm_isolated'] = statistics(query,gpu['query_norm'],positions)
checks['query_norm_native_tree'] = statistics(native_norm(residual,weights['norm_query']),gpu['query_norm'],positions)
for label, operation in (('gate_isolated',gate),('gate_native_tree',native_gate)):
    output, scalars = operation(gpu['key_norm'].reshape(-1,4,2560),gpu['query_norm'].reshape(-1,4,2560),gpu['value_proj_folded'])
    checks[label] = statistics(output,gpu['gated'],positions)
    if label=='gate_isolated':
        isolated_scalars = scalars.tolist()
checks['full_chain'] = statistics(gate(key,query,projected['value_proj'])[0],gpu['gated'],positions)
actual_key = gpu['key_norm'].reshape(-1,4,2560)
actual_query = gpu['query_norm'].reshape(-1,4,2560)
actual_value = gpu['value_proj_folded']
hybrids = {}
for label, operands in {
    'actual_all': (actual_key,actual_query,actual_value),
    'oracle_key_chain_only': (key,actual_query,actual_value),
    'oracle_query_norm_only': (actual_key,query,actual_value),
    'oracle_value_projection_only': (actual_key,actual_query,projected['value_proj']),
    'oracle_norm_given_actual_key': (norm(gpu['key_proj_folded'],weights['norm_key'],1e-6),actual_query,actual_value),
}.items():
    output,scalars = gate(*operands)
    hybrids[label] = dict(error_vs_captured=statistics(output,gpu['gated'],positions),
                         worst_position_136732_scalars=scalars[positions.index(136732)].tolist())
(root/'hybrid-attribution.json').write_text(json.dumps(hybrids,indent=2)+'\n')
report = dict(native_gpu_exactly_reproduces_all_original_outlier_gate_and_norm_values=True,
              fp32_output_control_gate_and_norm_also_exact=True,
              all_intermediate_fields_equal_across_output_modes={name:bool(np.array_equal(value,actual['fp32'][name])) for name,value in gpu.items()},
              checks=checks,positions=positions,isolated_gate_scalars=isolated_scalars,hybrid_attribution=hybrids,
              captured_stage_sha256=hashes,script_sha256=sha(Path(__file__)),
              scope='Standalone frozen-input replay at original native row offsets and shape. Does not run the original request or establish retrieval causality.')
(root/'gpu-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for key,value in checks.items():
    print(key,'L2',value['relative_l2'],'different',value['elements']-value['bitwise_equal_elements'])
