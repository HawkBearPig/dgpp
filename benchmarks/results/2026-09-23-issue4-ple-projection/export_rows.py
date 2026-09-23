#!/usr/bin/env python3
"""Read bounded PLE rows from the checksum-verified history archive."""
import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    selection = json.loads(args.positions.read_text())
    positions = selection['positions']
    assert positions == sorted(set(positions))
    assert min(positions) >= 0 and max(positions) < selection['tokens']
    entries = {row['path']: row for row in json.loads(args.manifest.read_text())['entries']}
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for rank in (0, 1):
        target = args.output / f'rank{rank}'
        target.mkdir()
        for field, width in [('ple_embedding', 1280), ('ple_residual_before', 10240),
                             ('ple_gv', 10240), ('ple_un', 10240)]:
            relative = f'rank{rank}/layer1/{field}.bin'
            source = args.archive / relative
            expected = entries[relative]
            before = source.stat()
            assert before.st_size == expected['bytes'] == selection['tokens'] * width * 2
            payload = bytearray()
            with source.open('rb') as stream:
                for row in positions:
                    stream.seek(row * width * 2)
                    chunk = stream.read(width * 2)
                    assert len(chunk) == width * 2
                    payload.extend(chunk)
            after = source.stat()
            assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
            (target / (field + '.bin')).write_bytes(payload)
            receipts.append(dict(source=relative, source_bytes=before.st_size,
                                 previously_verified_source_sha256=expected['sha256'],
                                 source_mtime_ns=before.st_mtime_ns, width=width,
                                 sample_bytes=len(payload), sample_sha256=hashlib.sha256(payload).hexdigest()))
        for field in ('ple.meta', 'ple_scale.bin'):
            relative = f'rank{rank}/layer1/{field}'
            payload = (args.archive / relative).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entries[relative]['sha256']
            (target / field).write_bytes(payload)
    result = dict(archive=str(args.archive), selection=selection, files=receipts,
                  full_archive_rehashed=False,
                  note='Reuses prior complete archive verification; current bounded reads check source sizes and stability plus all exported payload hashes.')
    (args.output / 'export.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(rows=len(positions), files=len(receipts), bytes=sum(x['sample_bytes'] for x in receipts))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
