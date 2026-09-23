#!/usr/bin/env python3
"""CPU checks of the independent conversion and streaming error accounting."""
import json
from pathlib import Path

import numpy as np
import torch

from replay import block_fp8, rb, statistics, sha
from stream_all import Error


rng = np.random.default_rng(913)
x = rng.normal(size=(256, 256)).astype(np.float32)
x[:128, :128] *= np.float32(.002)
x[128:, :128] = 0
x = rb(x)
w = torch.from_numpy(x).reshape(2, 128, 2, 128)
maximum = w.abs().amax(dim=(1, 3), keepdim=True)
scale = torch.where(maximum > 0, maximum / 448, torch.ones_like(maximum))
expected = ((w / scale).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16).float().reshape(256, 256).numpy()
assert np.array_equal(block_fp8(x), expected)
v = np.r_[rng.normal(size=10000).astype(np.float32), np.array([0., -0., 1e-38, 1e-40, -1e-40], np.float32)]
assert np.array_equal(rb(v).view('u4'), torch.from_numpy(v).to(torch.bfloat16).float().numpy().view('u4'))

# A defect in streaming statistics must not hide an injected local error.
reference = rb(rng.normal(size=(27, 40)).astype(np.float32))
actual = reference.copy()
actual[19, 3] += 2
actual[2, 10] -= 1
whole = statistics(reference, actual, list(range(27)))
streamed = Error()
for begin, end in ((0, 7), (7, 19), (19, 27)):
    streamed.add(reference[begin:end], actual[begin:end], begin)
result = streamed.result()
for key in ('elements', 'max_abs', 'bitwise_equal_elements', 'two_bf16_relative_mismatches', 'nonfinite'):
    assert result[key] == whole[key], key
assert abs(result['relative_l2'] - whole['relative_l2']) < 1e-15
assert result['two_bf16_relative_mismatches'] == 2
assert {row['position'] for row in result['worst_rows'][:2]} == {2, 19}
assert result['relative_l2'] > .002

report = dict(cpu_only=True, bf16_values=len(v), bf16_matches_torch_bits=True,
              fp8_matrix_values=x.size, fp8_block_dequant_matches_torch_bits=True,
              streaming_statistics_equal_to_whole=True, injected_errors_detected=2,
              cases=['normal values', 'varying block scale', 'zero block', 'subnormals', 'signed zero'],
              script_sha256=sha(Path(__file__)), oracle_sha256=sha(Path(__file__).with_name('replay.py')))
Path(__file__).with_name('oracle-preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
