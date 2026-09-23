#!/usr/bin/env python3
"""Compare production FP8 conversion against the independent real-weight oracle."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from replay import Weights, block_fp8, sha


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
library = ctypes.CDLL(str(args.library.resolve()))
pointer = ctypes.POINTER(ctypes.c_uint16)
library.encode.argtypes = [pointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, pointer]
library.encode.restype = None
weights = Weights(args.checkpoint)
checks = []
for name in ('key_proj', 'value_proj'):
    source = weights.read('model.language_model.layers.1.ple.' + name + '.weight')
    bits = np.ascontiguousarray((source.view('u4') >> 16).astype('<u2'))
    for rank in (0, 1):
        output = np.empty((len(source), 1280), dtype='<u2')
        address = bits.ctypes.data + rank * 1280 * 2
        library.encode(ctypes.cast(address, pointer), 2560, len(source), 1280, output.ctypes.data_as(pointer))
        expected = (block_fp8(source[:, rank*1280:(rank+1)*1280]).view('u4') >> 16).astype('<u2')
        mismatch = int(np.count_nonzero(output != expected))
        checks.append(dict(projection=name, rank=rank, elements=output.size, mismatches=mismatch,
                           dequantized_bf16_sha256=hashlib.sha256(output.tobytes()).hexdigest()))
assert all(x['mismatches'] == 0 for x in checks)
report = dict(checks=checks, all_exact=True, total_elements=sum(x['elements'] for x in checks),
              tensor_sha256=weights.hashes, native_library_sha256=sha(args.library), script_sha256=sha(Path(__file__)))
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
