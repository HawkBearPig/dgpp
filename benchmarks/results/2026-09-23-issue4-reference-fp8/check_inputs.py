#!/usr/bin/env python3
"""Audit captured reference inputs, PLE history, and checkpoint embeddings on CPU.

The trace saves all input IDs and positions, all two-token PLE contexts, and
embedding rows from the first two chunks plus the last row of each later call.
This checks those saved values against the original prompt and checkpoint; it
does not infer correctness of unrecorded embedding rows or downstream layers.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def captured(directory, field, dtype, original_shape, saved_shape):
    path = directory / f'layer-1_{field}.json'
    metadata = json.loads(path.read_text())
    data = path.with_suffix('.bin').read_bytes()
    expected_dtype = {'<i4': 'torch.int32', '<u2': 'torch.bfloat16'}[dtype]
    require(metadata['dtype'] == expected_dtype, f'{path}: dtype')
    require(metadata['original_shape'] == original_shape, f'{path}: original shape')
    require(metadata['saved_shape'] == saved_shape, f'{path}: saved shape')
    require(len(data) == metadata['bytes'], f'{path}: length')
    require(hashlib.sha256(data).hexdigest() == metadata['sha256'], f'{path}: SHA256')
    return np.frombuffer(data, dtype=dtype).reshape(saved_shape)


def embedding_tensor(model, hidden_size):
    name = 'model.language_model.embed_tokens.weight'
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    path = model / index['weight_map'][name]
    with path.open('rb') as stream:
        header_length, = struct.unpack('<Q', stream.read(8))
        metadata = json.loads(stream.read(header_length))[name]
    require(metadata['dtype'] == 'BF16', f'{path}: embedding dtype')
    require(metadata['shape'][1] == hidden_size, f'{path}: embedding width')
    start, end = metadata['data_offsets']
    require(end - start == 2 * int(np.prod(metadata['shape'])), f'{path}: tensor length')
    value = np.memmap(path, dtype='<u2', mode='r', offset=8 + header_length + start,
                      shape=tuple(metadata['shape']))
    digest = hashlib.sha256()
    for offset in range(0, len(value), 8192):
        digest.update(value[offset:offset + 8192].tobytes())
    return value, dict(name=name, file=path.name, dtype=metadata['dtype'],
                       shape=metadata['shape'], tensor_sha256=digest.hexdigest())


def audit(captures, prompt_ids, model):
    prompt = np.asarray(json.loads(prompt_ids.read_text()), dtype='<i8')
    require(len(prompt) == 261120, 'Expected the unchanged 261120-token issue #4 prompt')
    require(hashlib.sha256(prompt.tobytes()).hexdigest() ==
            'af58d17bb2d2c0308043a686e30baf1486b454da4f40059ee04306bfb29972be',
            'Original issue #4 prompt token SHA256 differs')
    config = json.loads((model / 'config.json').read_text())['text_config']
    eos, hidden, copies = (int(config[k]) for k in ('eos_token_id', 'hidden_size', 'hc_count'))
    require((eos, hidden, copies, config['ngram_size']) == (248044, 2560, 4, 3),
            'Unexpected checkpoint architecture or trained EOS')
    embedding, tensor_receipt = embedding_tensor(model, hidden)
    results = []
    baseline_sequence = None
    for rank in (0, 1):
        for request in (1, 2):
            root = captures / f'rank{rank}' / f'request{request}'
            directories = sorted(root.glob('position*'), key=lambda p: int(p.name[8:]))
            require(len(directories) == 297, f'{root}: expected 297 captured calls')
            sequence = []
            previous = [eos, eos]
            embedding_rows = 0
            for directory in directories:
                item = json.loads((directory / 'input.json').read_text())
                tokens, positions = item['tokens'], item['positions']
                position, rows = len(sequence), len(tokens)
                require(rows > 0 and item['position'] == position, f'{directory}: input continuity')
                require(directory.name == f'position{position}', f'{directory}: directory position')
                require(positions == list(range(position, position + rows)),
                        f'{directory}: token positions')
                expected_rows = min(2048, len(prompt) - position) if position < len(prompt) else 1
                require(rows == expected_rows, f'{directory}: prefill/decode row count')

                history = captured(directory, 'ngram_context', '<i4', [1, 2], [1, 2])
                require(history.tolist() == [previous], f'{directory}: PLE history differs')
                starts = [0, rows] if position in (0, 2048) else [rows]
                query = captured(directory, 'query_start_loc', '<i4', [2], [len(starts)])
                require(query.tolist() == starts, f'{directory}: query boundary differs')

                selected = tokens if position in (0, 2048) else tokens[-1:]
                saved = captured(directory, 'embedding', '<u2', [rows, hidden * copies],
                                 [len(selected), hidden * copies])
                require(all(0 <= token < len(embedding) for token in tokens),
                        f'{directory}: token outside checkpoint vocabulary')
                expected = embedding[selected]
                for copy in range(copies):
                    require(np.array_equal(saved[:, copy * hidden:(copy + 1) * hidden], expected),
                            f'{directory}: embedding copy {copy} differs from checkpoint')
                embedding_rows += len(selected)
                sequence.extend(tokens)
                previous = (previous + tokens)[-2:]

            sequence = np.asarray(sequence, dtype='<i8')
            require(np.array_equal(sequence[:len(prompt)], prompt), f'{root}: original prompt differs')
            if baseline_sequence is None:
                baseline_sequence = sequence
            require(np.array_equal(sequence, baseline_sequence), f'{root}: rank/request input differs')
            results.append(dict(rank=rank, request=request, calls=len(directories),
                                original_prompt_tokens_exact=len(prompt), consumed_tokens=len(sequence),
                                input_positions_contiguous=True, ngram_history_pairs_exact=len(directories),
                                query_boundary_samples_exact=len(directories),
                                embedding_rows_exact=embedding_rows, embedding_copies_per_row=copies,
                                final_consumed_token=int(sequence[-1]),
                                consumed_ids_int64le_sha256=hashlib.sha256(sequence.tobytes()).hexdigest()))
    return dict(passed=True, checkpoint=str(model), checkpoint_embedding=tensor_receipt,
                checkpoint_config_sha256=sha256(model / 'config.json'), trained_eos=eos,
                prompt_ids_file_sha256=sha256(prompt_ids),
                prompt_ids_int64le_sha256=hashlib.sha256(prompt.tobytes()).hexdigest(),
                capture_root=str(captures), ranks_and_requests=results,
                limits=['Embedding coverage is the first 4096 rows plus the final row of each later call.',
                        'Later query_start_loc captures contain only the final boundary.',
                        'This audits consumed inputs, not the mathematical correctness of generated answers.',
                        'N-gram context continuity does not validate all downstream PLE arithmetic.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('prompt_ids', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.captures, args.prompt_ids, args.model)
    result['script_sha256'] = sha256(Path(__file__))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
