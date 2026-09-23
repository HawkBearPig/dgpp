#!/usr/bin/env python3
"""Read-only verification against the snapshot's content-addressed blobs."""
import hashlib
import json
from pathlib import Path

snapshot = Path.home() / '.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47'
results = []
for path in sorted(snapshot.iterdir()):
    blob = path.resolve()
    expected = blob.name
    size = blob.stat().st_size
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
    if len(expected) == 40:
        digest.update(f'blob {size}\0'.encode())
    assert len(expected) in (40, 64), blob
    with blob.open('rb') as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    result = {'file': path.name, 'bytes': size, 'expected': expected,
              'actual': digest.hexdigest(), 'matches': digest.hexdigest() == expected}
    results.append(result)
    print(json.dumps(result), flush=True)
assert all(result['matches'] for result in results), 'Snapshot content mismatch'
