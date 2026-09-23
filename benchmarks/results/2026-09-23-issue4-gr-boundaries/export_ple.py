#!/usr/bin/env python3
"""Rehash the archived post-PLE residuals and export only their final row."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT.parent / '2026-09-23-issue4-ple-quantization/history-source-manifest.json').read_text())
expected = {entry['path']: entry for entry in manifest['entries']}
code = '''
import base64, hashlib, json
from pathlib import Path
root=Path('/home/stephen/dgpp-issue4-archive-20260923/history-captures')
result=[]
for rank in (0,1):
    name=f'rank{rank}/layer1/ple_residual_after.bin'
    path=root/name
    before=path.stat()
    with path.open('rb') as stream:
        checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        stream.seek(-10240*2,2)
        payload=stream.read(10240*2)
    after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    result.append(dict(source=name,bytes=before.st_size,sha256=checksum,payload=base64.b64encode(payload).decode()))
print(json.dumps(result))
'''
call = subprocess.run(['ssh','-o','BatchMode=yes','stephen@192.168.88.13','python3 -'],
                      input=code, capture_output=True, text=True, check=True, timeout=180)
rows = json.loads(call.stdout)
for rank, row in enumerate(rows):
    reference = expected[row['source']]
    assert row['bytes'] == reference['bytes'] == 261290 * 10240 * 2
    assert row['sha256'] == reference['sha256']
    payload = base64.b64decode(row.pop('payload'), validate=True)
    assert len(payload) == 10240 * 2
    row['export_sha256'] = hashlib.sha256(payload).hexdigest()
    (ROOT / f'raw/ple-after-rank{rank}.bin').write_bytes(payload)
assert rows[0]['export_sha256'] == rows[1]['export_sha256']
(ROOT / 'ple-export.json').write_text(json.dumps(dict(position=261289,full_source_rehash_matches=True,files=rows),indent=2)+'\n')
print('Exported both post-PLE final rows after full source rehash.',flush=True)
