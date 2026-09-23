#!/usr/bin/env python3
"""Replay the previously uncovered final-chunk QSA key projection and transform."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from replay import Weights, bf, rb, block_fp8, statistics, sha


def main(args):
    started = time.monotonic()
    weights = Weights(args.checkpoint)
    config = json.loads((args.checkpoint / 'config.json').read_text())['text_config']
    eps = config['rms_norm_eps']
    checks = []
    for layer in range(3, 48, 4):
        prefix = f'model.language_model.layers.{layer}.self_attn.'
        projection = weights.read(prefix + 'k_proj.weight')
        scale = weights.read(prefix + 'k_norm.weight')
        assert projection.shape == (512, 2560) and scale.shape == (256,)
        for rank in (0, 1):
            source = args.captures / f'rank{rank}/layer{layer}'
            metadata = (source / 'qsa.meta').read_text().split()
            assert list(map(int, metadata[:4])) == [1194, 12, 1, 256]
            rotary, last = int(metadata[-2]), int(metadata[-1])
            assert rotary == 64 and last == 261289
            positions = np.arange(last - 1194 + 1, last + 1, dtype=np.int64)
            x = bf(np.fromfile(source / 'attention_input.bin', dtype='<u2').reshape(1194, 2560))
            local = block_fp8(projection[rank*256:(rank+1)*256])
            raw = rb(x.astype(np.float64) @ local.astype(np.float64).T)
            rstd = (1 / np.sqrt(np.mean(raw.astype(np.float64)**2, axis=1, keepdims=True) + np.float32(eps))).astype(np.float32)
            normalized = rb((raw * rstd) * (np.float32(1) + scale))
            frequencies = np.fromfile(source / 'inv_freq.bin', dtype='<f4')
            assert frequencies.shape == (32,)
            angles = positions.astype(np.float32)[:, None] * frequencies[None, :]
            cosine, sine = rb(np.cos(angles.astype(np.float64))), rb(np.sin(angles.astype(np.float64)))
            expected = normalized.copy()
            first, second = normalized[:, :32], normalized[:, 32:64]
            expected[:, :32] = rb(rb(first * cosine) + rb(-second * sine))
            expected[:, 32:64] = rb(rb(second * cosine) + rb(first * sine))
            bits = np.fromfile(source / 'kn_tail.bin', dtype='<u2').reshape(1194, 256)
            cache_bits = np.fromfile(source / 'k_cache_tail.bin', dtype='<u2').reshape(1194, 256)
            assert np.array_equal(bits, cache_bits)
            result = statistics(expected, bf(bits), positions.tolist())
            assert result['nonfinite'] == 0
            fields = ['qsa.meta', 'attention_input.bin', 'inv_freq.bin', 'kn_tail.bin', 'k_cache_tail.bin']
            checks.append(dict(layer=layer, rank=rank, result=result, cache_tail_exact=True,
                               input_sha256={name: sha(source / name) for name in fields}))
        print('Layer', layer, 'key transform L2', [row['result']['relative_l2'] for row in checks[-2:]], flush=True)
    report = dict(completed=True, layer_rank_comparisons=len(checks), rows_per_layer_rank=1194,
                  total_elements=sum(row['result']['elements'] for row in checks), checks=checks,
                  maximum_relative_l2=max(row['result']['relative_l2'] for row in checks),
                  nonfinite=sum(row['result']['nonfinite'] for row in checks),
                  tensor_sha256=weights.hashes, seconds=time.monotonic()-started,
                  script_sha256=sha(Path(__file__)), oracle_sha256=sha(Path(__file__).with_name('replay.py')),
                  scope='Checkpoint FP8 key projection, grouped norm and RoPE for all final-chunk rows at 24 layer/rank pairs. Starts from captured attention inputs; earlier inputs and cache rows are not replayed.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Complete:', report['total_elements'], 'elements; max L2', report['maximum_relative_l2'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
