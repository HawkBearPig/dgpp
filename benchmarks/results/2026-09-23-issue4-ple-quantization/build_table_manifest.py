#!/usr/bin/env python3
"""Bind the BF16 table diagnostic to verified files and the original hash geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


PREFIX = 'model.language_model.layers.1.ple.ple_embedding.'


class Tensors:
    def __init__(self, directory, index):
        self.directory = directory
        self.index = json.loads(index.read_text())['weight_map']
        self.headers = {}

    def metadata(self, name):
        path = self.directory / self.index[name]
        if path not in self.headers:
            with path.open('rb') as stream:
                length, = struct.unpack('<Q', stream.read(8))
                if length > 1024 * 1024:
                    raise ValueError('Unexpected safetensors header length')
                self.headers[path] = (8 + length, json.loads(stream.read(length)))
        start, header = self.headers[path]
        entry = header[name]
        return path, start + entry['data_offsets'][0], entry

    def small_int64(self, name):
        path, start, entry = self.metadata(name)
        length = entry['data_offsets'][1] - entry['data_offsets'][0]
        if entry['dtype'] != 'I64' or length > 4096 or length % 8:
            raise ValueError('Unexpected hash buffer metadata: ' + name)
        with path.open('rb') as stream:
            stream.seek(start)
            data = stream.read(length)
        return list(struct.unpack('<' + 'q' * (length // 8), data))


def main(args):
    verification = json.loads(args.verification.read_text())
    if not verification['all_verified'] or len(verification['files']) != 33:
        raise ValueError('The 33 source files must pass size/SHA256 verification first')
    verified = {row['name']: row for row in verification['files']}
    for name, row in verified.items():
        if not row['verified'] or (args.directory / name).stat().st_size != row['bytes']:
            raise ValueError('Source file differs from verification receipt: ' + name)
    source = Tensors(args.directory, args.source_index)
    original = Tensors(args.original_checkpoint, args.original_checkpoint / 'model.safetensors.index.json')
    buffers = {}
    for key, suffix in (('multipliers', 'layer_multipliers'), ('head_vocab', 'ngram_heads_vocab_sizes'),
                        ('head_offset', 'ngram_heads_offsets')):
        value = source.small_int64(PREFIX + suffix)
        if value != original.small_int64(PREFIX + suffix):
            raise ValueError('BF16 and original checkpoint hash buffers differ: ' + suffix)
        buffers[key] = value
    parts = []
    for shard in range(128):
        name = PREFIX + f'ngram_embedding.shard_{shard}.weight'
        path, offset, entry = source.metadata(name)
        _, _, other = original.metadata(name)
        if entry['dtype'] != 'BF16' or entry['shape'] != other['shape'] or entry['shape'][1] != 160:
            raise ValueError('BF16 table geometry differs: ' + name)
        rows, width = entry['shape']
        if offset + rows * width * 2 > path.stat().st_size:
            raise ValueError('BF16 table extends beyond source file')
        parts.append(dict(path=str(path), data_begin=offset, rows=rows,
                          source_file_sha256=verified[path.name]['sha256']))
    capacity = parts[0]['rows']
    if any(part['rows'] != capacity for part in parts[:-1]):
        raise ValueError('Nonfinal PLE table shards differ in row count')
    total_rows = sum(part['rows'] for part in parts)
    ranks = []
    for rank in (0, 1):
        begin = buffers['head_offset'][rank * 8]
        end = buffers['head_offset'][8] if rank == 0 else total_rows
        names = sorted({Path(part['path']).name for shard, part in enumerate(parts)
                        if shard * capacity < end and shard * capacity + part['rows'] > begin})
        ranks.append(dict(rank=rank, row_begin=begin, row_count=end - begin,
                          files=[verified[name] for name in names],
                          file_bytes=sum(verified[name]['bytes'] for name in names)))
    report = dict(source_checkpoint=verification['checkpoint'], dtype='BF16', head_dim=160,
                  capacity=capacity, **buffers, parts=parts, rank_files=ranks,
                  verification_sha256=hashlib.sha256(args.verification.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(parts=len(parts), capacity=capacity, total_rows=total_rows,
                         rank_file_bytes=[row['file_bytes'] for row in ranks]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source-index', type=Path, required=True)
    parser.add_argument('--original-checkpoint', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
