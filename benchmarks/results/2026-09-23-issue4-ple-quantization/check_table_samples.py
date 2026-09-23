#!/usr/bin/env python3
"""Compare bounded BF16 table samples with the resident FP8 checkpoint on CPU.

Downloads only explicit byte ranges from a pinned public safetensors snapshot.
Rejects servers that ignore Range. This is a sampled quantization audit, not a
full-checkpoint integrity check or a retrieval accuracy experiment.
"""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

BF16_REPO = 'Qwen/Qwen3.8-Flash-Next'
BF16_REV = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
PREFIX = 'model.language_model.layers.1.ple.'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


class LocalCheckpoint:
    def __init__(self, root):
        self.root = root
        self.index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}

    def tensor(self, name):
        path = self.root / self.index[name]
        if path not in self.headers:
            with path.open('rb') as stream:
                length, = struct.unpack('<Q', stream.read(8))
                self.headers[path] = (8 + length, json.loads(stream.read(length)))
        offset, header = self.headers[path]
        entry = header[name]
        dtype = {'BF16': '<u2', 'F8_E4M3': 'u1', 'I64': '<i8'}[entry['dtype']]
        return np.memmap(path, dtype=dtype, mode='r', shape=tuple(entry['shape']),
                         offset=offset + entry['data_offsets'][0])


class RemoteCheckpoint:
    def __init__(self, index_path, api_path, cache):
        self.index = json.loads(index_path.read_text())['weight_map']
        api = json.loads(api_path.read_text())
        require(api['sha'] == BF16_REV, 'BF16 metadata revision differs')
        self.files = {entry['rfilename']: entry for entry in api['siblings']}
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.headers = {}
        self.receipts = []

    def read(self, filename, start, length):
        require(0 < length <= 1024 * 1024, 'Range exceeds the 1 MiB read limit')
        end = start + length - 1
        size = self.files[filename]['size']
        require(0 <= start <= end < size, 'Range exceeds checkpoint file')
        stem = f'{filename}.{start}-{end}'
        path = self.cache / (stem + '.bin')
        receipt_path = self.cache / (stem + '.json')
        url = f'https://huggingface.co/{BF16_REPO}/resolve/{BF16_REV}/{filename}'
        if path.exists() and receipt_path.exists():
            data = path.read_bytes()
            receipt = json.loads(receipt_path.read_text())
            require(len(data) == length and digest(data) == receipt['sha256'], 'Cached range differs')
            require(receipt['url'] == url and receipt['start'] == start and receipt['bytes'] == length,
                    'Cached range identity differs')
        else:
            for attempt in range(3):
                try:
                    request = Request(url + f'?download=true&issue4_range={start}-{end}',
                                      headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                    with urlopen(request, timeout=30) as response:
                        require(response.status == 206, f'Range request status {response.status}')
                        require(response.headers.get('Content-Range') == f'bytes {start}-{end}/{size}',
                                'Server returned an unexpected byte range')
                        data = response.read(length + 1)
                        require(len(data) == length, 'Range length differs')
                    break
                except (URLError, TimeoutError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(1)
            receipt = dict(url=url, start=start, bytes=length, sha256=digest(data),
                           source_file_bytes=size,
                           source_file_lfs_sha256=self.files[filename]['lfs']['sha256'])
            path.write_bytes(data)
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        self.receipts.append(receipt)
        return data

    def metadata(self, name):
        filename = self.index[name]
        if filename not in self.headers:
            length, = struct.unpack('<Q', self.read(filename, 0, 8))
            header = json.loads(self.read(filename, 8, length))
            self.headers[filename] = (8 + length, header)
        offset, header = self.headers[filename]
        return filename, offset, header[name]

    def rows(self, name, start, count):
        filename, offset, entry = self.metadata(name)
        dtype = {'BF16': '<u2', 'I64': '<i8'}[entry['dtype']]
        shape = entry['shape']
        require(start >= 0 and start + count <= shape[0], 'Tensor row range differs')
        row_bytes = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
        data = self.read(filename, offset + entry['data_offsets'][0] + start * row_bytes,
                         count * row_bytes)
        return np.frombuffer(data, dtype=dtype).reshape([count] + shape[1:])


def main(args):
    local = LocalCheckpoint(args.fp8)
    remote = RemoteCheckpoint(args.bf16_metadata / 'model-index.json',
                              args.bf16_metadata / 'model-api.json', args.cache)
    scale_bits = local.tensor(PREFIX + 'ple_embedding.ngram_embedding.weight_scale')
    scale = float(bf16(scale_bits).item())
    require(np.isfinite(scale) and scale > 0, 'Invalid checkpoint table scale')
    codes = np.arange(127, dtype=np.uint16)
    exponents, mantissas = (codes >> 3) & 15, codes & 7
    positive = np.where(exponents == 0, mantissas * 2.0**-9,
                        (1 + mantissas / 8) * np.exp2(exponents.astype(np.int32) - 7)).astype(np.float32)
    samples, shared = [], []
    sum_error_sq = sum_source_sq = sum_code_source = sum_code_sq = 0.0
    for shard in (0, 1, 31, 63, 64, 95, 126, 127):
        name = PREFIX + f'ple_embedding.ngram_embedding.shard_{shard}.weight'
        encoded = local.tensor(name)
        _, _, metadata = remote.metadata(name)
        require(metadata['dtype'] == 'BF16' and metadata['shape'] == list(encoded.shape),
                f'{name}: source tensor differs')
        for start in (0, len(encoded) // 2, len(encoded) - 64):
            original_bits = remote.rows(name, start, 64)
            original = bf16(original_bits).astype(np.float64)
            stored = np.asarray(encoded[start:start + 64])
            require(np.all((stored & 127) < 127), f'{name}: nonfinite FP8 code')
            decoded = positive[stored & 127].astype(np.float64) * np.where(stored & 128, -1, 1)
            dequantized = decoded * scale
            # Independent round-to-nearest/even FP8 encoder using the stored scale.
            magnitude = np.abs(original) / scale
            high = np.clip(np.searchsorted(positive, magnitude), 0, 126)
            low = np.maximum(high - 1, 0)
            lower_distance = np.abs(magnitude - positive[low])
            upper_distance = np.abs(positive[high] - magnitude)
            choose_low = (lower_distance < upper_distance) | ((lower_distance == upper_distance) & ((low & 1) == 0))
            expected = np.where(choose_low, low, high).astype(np.uint8) | (np.signbit(original).astype(np.uint8) << 7)
            error = dequantized - original
            error_sq = float(np.sum(error * error))
            source_sq = float(np.sum(original * original))
            sum_error_sq += error_sq
            sum_source_sq += source_sq
            sum_code_source += float(np.sum(decoded * original))
            sum_code_sq += float(np.sum(decoded * decoded))
            sample = dict(shard=shard, start_row=start, rows=64, elements=int(stored.size),
                          source_bf16_sha256=digest(original_bits.tobytes()),
                          checkpoint_fp8_sha256=digest(stored.tobytes()),
                          relative_l2=float(np.sqrt(error_sq / source_sq)),
                          maximum_absolute_error=float(np.max(np.abs(error))),
                          exact_requantized_codes=int(np.count_nonzero(expected == stored)),
                          codes_more_than_one_step_apart=int(np.count_nonzero(np.abs((expected & 127).astype(int) - (stored & 127).astype(int)) > 1)),
                          nonzero_sign_disagreements=int(np.count_nonzero((np.signbit(original) != ((stored & 128) != 0)) & (original != 0) & ((stored & 127) != 0))),
                          source_zeros=int(np.count_nonzero(original == 0)),
                          decoded_zeros=int(np.count_nonzero(decoded == 0)))
            samples.append(sample)
        print(f'Compared shard {shard}: 192 BF16/FP8 rows', flush=True)

    # These unchanged buffers/projections bind the two checkpoints more tightly
    # than the tensor names alone. They are sampled, not a full weight comparison.
    for suffix in ('ple_embedding.layer_multipliers', 'ple_embedding.ngram_heads_vocab_sizes',
                   'ple_embedding.ngram_heads_offsets', 'norm_key.weight', 'norm_query.weight',
                   'norm_conv.weight', 'key_proj.weight', 'value_proj.weight'):
        name = PREFIX + suffix
        stored = local.tensor(name)
        _, _, metadata = remote.metadata(name)
        require(list(stored.shape) == metadata['shape'], f'{name}: shared shape differs')
        count = min(len(stored), 4) if stored.ndim > 1 else len(stored)
        source = remote.rows(name, 0, count)
        shared.append(dict(name=name, rows=count, equal=np.array_equal(source, stored[:count]),
                           bf16_source_bytes_sha256=digest(source.tobytes()),
                           fp8_checkpoint_bytes_sha256=digest(stored[:count].tobytes())))

    total = sum(row['elements'] for row in samples)
    result = dict(bf16_checkpoint=f'{BF16_REPO}@{BF16_REV}', fp8_checkpoint=str(args.fp8),
                  table_scale=scale, scale_bf16_bits=int(scale_bits.item()),
                  table_rows=sum(row['rows'] for row in samples), table_elements=total,
                  relative_l2=float(np.sqrt(sum_error_sq / sum_source_sq)),
                  fitted_scale=sum_code_source / sum_code_sq,
                  fitted_to_stored_scale_ratio=sum_code_source / sum_code_sq / scale,
                  exact_requantized_codes=sum(row['exact_requantized_codes'] for row in samples),
                  codes_more_than_one_step_apart=sum(row['codes_more_than_one_step_apart'] for row in samples),
                  nonzero_sign_disagreements=sum(row['nonzero_sign_disagreements'] for row in samples),
                  samples=samples, shared_tensors=shared, ranges=remote.receipts,
                  bytes_read=sum(row['bytes'] for row in remote.receipts),
                  source_script_sha256=digest(Path(__file__).read_bytes()),
                  limits=['Only the listed row blocks and shared tensor prefixes were sampled.',
                          'Range hashes identify retained bytes; complete BF16 file hashes were not verified.',
                          'Small requantization differences can follow from rounding the stored global scale.',
                          'This does not determine whether quantization causes the retrieval failure.'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('samples', 'ranges', 'shared_tensors')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp8', type=Path, required=True)
    parser.add_argument('--bf16-metadata', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
