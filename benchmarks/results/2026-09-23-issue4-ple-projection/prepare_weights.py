#!/usr/bin/env python3
"""Prepare the small, independently reconstructed PLE matrices for CPU replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from replay import Weights, block_fp8, sha


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Weights(args.checkpoint)
    prefix = 'model.language_model.layers.1.ple.'
    matrices = {}
    for name in ('key_proj', 'value_proj'):
        weight = source.read(prefix + name + '.weight')
        for rank in (0, 1):
            matrices[name + str(rank)] = block_fp8(weight[:, rank*1280:(rank+1)*1280])
    for name in ('norm_key', 'norm_query', 'norm_conv'):
        matrices[name] = source.read(prefix + name + '.weight')
    np.savez(args.output, **matrices)
    args.output.with_suffix('.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        tensor_sha256=source.hashes, bundle_sha256=sha(args.output),
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key,value in matrices.items()}), indent=2)+'\n')
    print('Prepared', args.output.stat().st_size, 'bytes')
