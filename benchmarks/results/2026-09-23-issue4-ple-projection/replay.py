#!/usr/bin/env python3
"""Independent CPU checkpoint-to-PLE-gate replay on actual captured inputs.

The oracle implements block FP8 conversion, FP64 projection products, explicit
BF16 rank partials/folds, grouped normalization and the staged PLE gate. It does
not call DGPP operators or its host reference. Captured gate outputs are used
only for comparison, and separately for an isolated convolution-norm check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bf(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def rb(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return bf((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16)


def block_fp8(weight):
    nr, nc = weight.shape
    assert nr % 128 == nc % 128 == 0
    blocks = np.ascontiguousarray(weight).reshape(nr // 128, 128, nc // 128, 128)
    maximum = np.abs(blocks).max(axis=(1, 3), keepdims=True)
    scale = np.where(maximum > 0, maximum / np.float32(448), np.float32(1))
    scaled = blocks / scale
    _, exponent = np.frexp(np.abs(scaled))
    step = np.exp2(np.maximum(exponent - 4, -9)).astype(np.float32)
    codes = np.clip(np.rint(scaled / step) * step, -448, 448)
    return rb(codes * scale).reshape(nr, nc)


class Weights:
    def __init__(self, path):
        self.path = path
        self.index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers, self.hashes = {}, {}

    def read(self, name):
        path = self.path / self.index[name]
        if path not in self.headers:
            with path.open('rb') as stream:
                length, = struct.unpack('<Q', stream.read(8))
                assert length < 64 * 1024 * 1024
                self.headers[path] = (8 + length, json.loads(stream.read(length)))
        base, header = self.headers[path]
        item = header[name]
        assert item['dtype'] == 'BF16'
        bits = np.memmap(path, mode='r', dtype='<u2', offset=base + item['data_offsets'][0], shape=tuple(item['shape']))
        self.hashes[name] = hashlib.sha256(bits.tobytes()).hexdigest()
        return bf(bits)


def norm(values, weight, eps):
    grouped = values.reshape(-1, 4, 2560)
    variance = np.mean(grouped.astype(np.float64) ** 2, axis=-1, keepdims=True)
    rstd = (1 / np.sqrt(variance + np.float32(eps))).astype(np.float32)
    return rb((grouped * rstd) * (np.float32(1) + weight.reshape(4, 2560)))


def gate(key, query, value):
    products = rb(key * query)
    dot = rb(products.sum(axis=-1, dtype=np.float64))
    scaled = rb(dot / np.float32(np.sqrt(2560.0)))
    rooted = rb(np.sqrt(np.maximum(np.abs(scaled), np.float32(1e-6)))) * np.sign(scaled)
    sigmoid = rb(1 / (1 + np.exp(-rooted.astype(np.float64))))
    return rb(sigmoid[..., None] * value[:, None, :]), sigmoid


def statistics(expected, actual, positions):
    expected = expected.astype(np.float64).reshape(len(positions), -1)
    actual = actual.astype(np.float64).reshape(expected.shape)
    error = np.abs(expected - actual)
    reference_energy = np.sum(expected ** 2, axis=1)
    actual_energy = np.sum(actual ** 2, axis=1)
    error_energy = np.sum(error ** 2, axis=1)
    relative = np.sqrt(error_energy / np.maximum(np.maximum(reference_energy, actual_energy), 1e-60))
    threshold = np.maximum(1e-7, np.maximum(np.abs(expected), np.abs(actual)) * (2 / 128))
    mismatches = error > threshold
    worst = np.argsort(relative)[-10:][::-1]
    return dict(elements=error.size, relative_l2=float(np.sqrt(error_energy.sum() / max(reference_energy.sum(), actual_energy.sum(), 1e-60))),
                max_abs=float(error.max()), bitwise_equal_elements=int((expected == actual).sum()),
                two_bf16_relative_mismatches=int(mismatches.sum()),
                mismatch_fraction=float(mismatches.mean()), nonfinite=int((~np.isfinite(expected) | ~np.isfinite(actual)).sum()),
                worst_rows=[dict(position=positions[i], relative_l2=float(relative[i]), max_abs=float(error[i].max()),
                                 mismatches=int(mismatches[i].sum())) for i in worst])


def main(args):
    start = time.monotonic()
    export = json.loads((args.rows / 'export.json').read_text())
    positions = export['selection']['positions']
    arrays = {}
    for entry in export['files']:
        rank, _, name = entry['source'].split('/')
        path = args.rows / rank / name
        assert path.stat().st_size == entry['sample_bytes'] and sha(path) == entry['sample_sha256']
        arrays[(rank, name)] = bf(np.fromfile(path, dtype='<u2').reshape(len(positions), entry['width']))
    for field in ('ple_residual_before.bin', 'ple_gv.bin', 'ple_un.bin'):
        assert np.array_equal(arrays[('rank0', field)], arrays[('rank1', field)])
    weights = Weights(args.checkpoint)
    config = json.loads((args.checkpoint / 'config.json').read_text())['text_config']
    assert config['hidden_size'] == 2560 and config['hc_count'] == 4 and config['ple_embed_dim'] == 2560
    prefix = 'model.language_model.layers.1.ple.'
    projections = {}
    projection_receipts = []
    for name, width in (('key_proj', 10240), ('value_proj', 2560)):
        full = weights.read(prefix + name + '.weight')
        assert full.shape == (width, 2560)
        parts = []
        for rank in (0, 1):
            matrix = block_fp8(full[:, rank * 1280:(rank + 1) * 1280])
            embedding = arrays[(f'rank{rank}', 'ple_embedding.bin')]
            partial = rb(embedding.astype(np.float64) @ matrix.astype(np.float64).T)
            parts.append(partial)
            projection_receipts.append(dict(projection=name, rank=rank,
                                            reconstructed_dequant_weight_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
                                            partial_sha256=hashlib.sha256(partial.tobytes()).hexdigest()))
            print(name, 'rank', rank, 'projected', len(positions), 'rows', flush=True)
        projections[name] = rb(parts[0] + parts[1])
    eps = config['rms_norm_eps']
    key = norm(projections['key_proj'], weights.read(prefix + 'norm_key.weight'), eps)
    query = norm(arrays[('rank0', 'ple_residual_before.bin')], weights.read(prefix + 'norm_query.weight'), eps)
    gated, gates = gate(key, query, projections['value_proj'])
    conv_weight = weights.read(prefix + 'norm_conv.weight')
    normalized = norm(gated, conv_weight, eps)
    actual_gate = arrays[('rank0', 'ple_gv.bin')]
    actual_norm = arrays[('rank0', 'ple_un.bin')]
    checks = dict(checkpoint_to_gate=statistics(gated, actual_gate, positions),
                  checkpoint_to_conv_norm=statistics(normalized, actual_norm, positions),
                  isolated_conv_norm=statistics(norm(actual_gate, conv_weight, eps), actual_norm, positions))
    assert all(x['nonfinite'] == 0 for x in checks.values())
    result = dict(rows=len(positions), checks=checks, rank_inputs_and_outputs_equal=True,
                  projections=projection_receipts, tensor_sha256=weights.hashes,
                  gate_range=[float(gates.min()), float(gates.max())],
                  elapsed_seconds=time.monotonic()-start, script_sha256=sha(Path(__file__)),
                  export_receipt_sha256=sha(args.rows / 'export.json'),
                  scope='Independent checkpoint-to-gate chain on 774 captured rows. Starts from captured table embeddings and pre-PLE residuals. Does not validate earlier residual construction, unsampled rows, or end-to-end retrieval.',
                  comparison_policy='Report complete numerical errors without tuning a tolerance to this fixture. Existing isolated PLE gate test uses 0.002 L2 and 1% two-BF16-relative mismatches, but does not include this sharded FP8 projection chain.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(rows=len(positions), checks=checks, elapsed_seconds=result['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
