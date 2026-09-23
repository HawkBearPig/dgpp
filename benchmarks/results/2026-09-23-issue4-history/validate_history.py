#!/usr/bin/env python3
"""Validate complete capture coverage and independently replay PLE lookup."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parent
CAP = ROOT / 'raw/captures'
PREVIOUS = ROOT.parent / '2026-09-23-issue4-operators'
MODEL = Path('/home/stephen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47')
tokens = np.asarray(json.loads((PREVIOUS / 'raw/forced-prefix.ids.json').read_text()), dtype='<i8')
assert hashlib.sha256(tokens.tobytes()).hexdigest() == 'fe207c412e6a70d70b00bbc3c5995488ed12ce964c4ac7eb90933ab61e82785d'
chunks = list(range(0, len(tokens), 2048))
coverage = []
for rank in range(2):
    p = CAP / f'rank{rank}/layer0'
    assert np.array_equal(np.fromfile(p / 'token_ids.bin', dtype='<i8'), tokens)
    assert np.array_equal(np.fromfile(p / 'positions.bin', dtype='<i8'), np.arange(len(tokens)))
    assert np.all(np.fromfile(p / 'request_ids.bin', dtype='<i4') == 0)
    for layer in range(48):
        p = CAP / f'rank{rank}/layer{layer}'
        checks = [json.loads(line) for line in (p / 'checks.jsonl').read_text().splitlines()]
        names = {'chunk', 'k_all_visible_exact', 'v_all_visible_exact', 'old_index_exact'} if layer % 4 == 3 else {'chunk', 'recurrent_before_exact', 'conv_before_exact', 'conv_tail_exact'}
        if layer == 1:
            names |= {'ple_conv_before_exact', 'ple_conv_tail_exact'}
        assert {c['check'] for c in checks} == names
        for name in names:
            subset = [c for c in checks if c['check'] == name]
            assert [c['position'] for c in subset] == chunks
            assert all(c['rows'] == min(2048, len(tokens) - c['position']) for c in subset)
        residuals = np.fromfile(p / 'last_residual.bin', dtype='<u2').reshape(len(chunks), 10240)
        old = np.fromfile(PREVIOUS / f'raw/captures/rank{rank}/layer{layer}/post_mlp_residual.bin', dtype='<u2')
        assert np.array_equal(residuals[-1], old)
        coverage.append({'rank': rank, 'layer': layer, 'checks': len(checks), 'chunks': len(chunks), 'final_residual_matches_prior_capture': True})

headers = {}
index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']


def tensor(name, dtype):
    path = MODEL / index[name]
    if path not in headers:
        with path.open('rb') as f:
            n, = struct.unpack('<Q', f.read(8))
            headers[path] = 8+n, json.loads(f.read(n))
    offset, metadata = headers[path]
    entry = metadata[name]
    return np.memmap(path, dtype=dtype, mode='r', offset=offset+entry['data_offsets'][0], shape=tuple(entry['shape']))


def bf(x):
    return (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)


def rb(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')


# Independent E4M3FN code table, including signed zero and finite exponent 15.
codes = np.arange(256, dtype=np.uint16)
exponents, mantissas = (codes >> 3) & 15, codes & 7
lut = np.where(exponents == 0, mantissas * (2.0**-9), (1 + mantissas / 8) * np.exp2(exponents.astype(np.int32) - 7)).astype(np.float32)
lut *= np.where(codes & 128, -1, 1).astype(np.float32)
lut[(codes & 127) == 127] = np.nan
prefix = 'model.language_model.layers.1.ple.ple_embedding.'
mult = np.asarray(tensor(prefix + 'layer_multipliers', '<i8')).reshape(-1)
vocab = np.asarray(tensor(prefix + 'ngram_heads_vocab_sizes', '<i8')).reshape(-1)
offsets = np.asarray(tensor(prefix + 'ngram_heads_offsets', '<i8')).reshape(-1)
table_scale = bf(tensor(prefix + 'ngram_embedding.weight_scale', '<u2')).item()
parts = [tensor(prefix + f'ngram_embedding.shard_{s}.weight', 'u1') for s in range(128)]
ends = np.cumsum([len(part) for part in parts])
starts = np.r_[0, ends[:-1]]
model_eos = json.loads((MODEL / 'config.json').read_text())['text_config']['eos_token_id']
eos = int((CAP / 'rank0/layer1/ple.meta').read_text().split()[4])
prev1 = np.r_[eos, tokens[:-1]].astype('<i8')
prev2 = np.r_[eos, eos, tokens[:-2]].astype('<i8')
prev2 = np.where(prev1 == eos, eos, prev2)
with np.errstate(over='ignore'):
    mixed2 = tokens.astype(np.uint64) * np.uint64(mult[0]) ^ prev1.astype(np.uint64) * np.uint64(mult[1])
    mixed3 = mixed2 ^ prev2.astype(np.uint64) * np.uint64(mult[2])
expected_ids = np.empty((len(tokens), 16), dtype=np.int32)
for h in range(16):
    mixed = mixed2 if h < 8 else mixed3
    expected_ids[:, h] = mixed.view(np.int64) % vocab[h] + offsets[h]
# Also compare the runtime hash IDs with the trained model's EOS policy.
model_prev1 = np.r_[model_eos, tokens[:-1]].astype('<i8')
model_prev2 = np.r_[model_eos, model_eos, tokens[:-2]].astype('<i8')
model_prev2 = np.where(model_prev1 == model_eos, model_eos, model_prev2)
with np.errstate(over='ignore'):
    model_mixed2 = tokens.astype(np.uint64) * np.uint64(mult[0]) ^ model_prev1.astype(np.uint64) * np.uint64(mult[1])
    model_mixed3 = model_mixed2 ^ model_prev2.astype(np.uint64) * np.uint64(mult[2])
model_ids = np.empty_like(expected_ids)
for h in range(16):
    mixed = model_mixed2 if h < 8 else model_mixed3
    model_ids[:, h] = mixed.view(np.int64) % vocab[h] + offsets[h]
different = np.flatnonzero(np.any(model_ids != expected_ids, axis=1))
eos_discrepancy = {'model_eos': model_eos, 'runtime_eos': eos,
                   'different_positions': different.tolist(),
                   'different_ids': int(np.count_nonzero(model_ids != expected_ids))}
print('Trained EOS discrepancy:', eos_discrepancy, flush=True)
ple = []
for rank in range(2):
    p = CAP / f'rank{rank}/layer1'
    heads, begin, local, dim, captured_eos, *_ = map(int, (p / 'ple.meta').read_text().split())
    assert captured_eos == eos and heads == 16
    assert np.array_equal(np.fromfile(p / 'ple_multipliers.bin', dtype='<i8'), mult)
    assert np.array_equal(np.fromfile(p / 'ple_vocab.bin', dtype='<i8'), vocab)
    assert np.array_equal(np.fromfile(p / 'ple_offset.bin', dtype='<i8'), offsets)
    assert np.fromfile(p / 'ple_scale.bin', dtype='<f4')[0] == table_scale
    ids = np.fromfile(p / 'ple_ids.bin', dtype='<i4').reshape(-1, heads)
    assert np.array_equal(ids, expected_ids)
    actual = np.memmap(p / 'ple_embedding.bin', dtype='<u2', mode='r', shape=(len(tokens), local, dim))
    mismatch = nonfinite = 0
    digest = hashlib.sha256()
    for start in chunks:
        wanted = expected_ids[start:start+2048, begin:begin+local].reshape(-1)
        shards = np.searchsorted(ends, wanted, side='right')
        rows = wanted - starts[shards]
        data = np.empty((len(wanted), dim), dtype=np.uint8)
        for shard in np.unique(shards):
            mask = shards == shard
            data[mask] = parts[shard][rows[mask]]
        dequant = lut[data] * np.float32(table_scale)
        nonfinite += int((~np.isfinite(dequant)).sum())
        expected = rb(dequant).reshape(-1, local, dim)
        digest.update(expected.tobytes())
        mismatch += int((expected != actual[start:start+len(expected)]).sum())
    ple.append({'rank': rank, 'tokens': len(tokens), 'hash_ids_exact': True, 'embedding_mismatches': mismatch, 'nonfinite': nonfinite, 'expected_embedding_sha256': digest.hexdigest()})
    print('PLE lookup', ple[-1], flush=True)
    assert mismatch == nonfinite == 0
summary = {'eos_discrepancy': eos_discrepancy, 'coverage': coverage, 'total_live_checks': sum(x['checks'] for x in coverage), 'tokens_exact_each_rank': True, 'all_chunk_positions_exact': True, 'final_layer_residuals_match_previous': True, 'ple': ple}
(ROOT / 'raw/history-validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print('History invariants and runtime PLE lookups verified; trained EOS discrepancy recorded', flush=True)
