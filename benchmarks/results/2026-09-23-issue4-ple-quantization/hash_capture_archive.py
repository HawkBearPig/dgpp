#!/usr/bin/env python3
"""Hash retained captures, or verify a relocated copy against that manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def main(args):
    paths = sorted(path for path in args.root.rglob('*') if path.is_file())
    if not paths or any(path.is_symlink() for path in paths):
        raise ValueError('Expected a nonempty capture directory of regular files')

    def check(path):
        before = path.stat()
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Capture changed while hashing: ' + str(path))
        return dict(path=str(path.relative_to(args.root)), bytes=after.st_size, sha256=checksum)

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, paths))
    report = dict(root=str(args.root), files=len(rows), bytes=sum(row['bytes'] for row in rows), entries=rows)
    if args.verify:
        original = json.loads(args.verify.read_text())
        if rows != original['entries']:
            raise ValueError('Archive file set, size or SHA256 differs from source manifest')
        report['matches_source'] = True
        report['source_manifest_sha256'] = hashlib.sha256(args.verify.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'entries'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', type=Path)
    main(parser.parse_args())
