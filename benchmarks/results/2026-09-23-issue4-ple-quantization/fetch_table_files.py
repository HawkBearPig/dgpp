#!/usr/bin/env python3
"""Fetch and verify only the BF16 snapshot files containing PLE table tensors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(args):
    manifest = json.loads(args.manifest.read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    required = sum(row['bytes'] for row in manifest['files'] if not (args.destination / row['name']).exists())
    if shutil.disk_usage(args.destination).free < required + 10 * 1024**3:
        raise RuntimeError('Insufficient disk space for the selected table files plus 10 GiB reserve')
    print(json.dumps(dict(pid=os.getpid(), files=len(manifest['files']), bytes=required,
                          destination=str(args.destination))), flush=True)

    def fetch(row):
        path = args.destination / row['name']
        partial = path.with_suffix(path.suffix + '.partial')
        began = time.monotonic()
        print('start', row['name'], flush=True)
        if not path.exists():
            proxy_args = ['--proxy', args.proxy] if args.proxy else []
            subprocess.run(['curl', *proxy_args, '--location', '--fail', '--silent', '--show-error',
                            '--retry', '4', '--retry-delay', '2', '--connect-timeout', '15',
                            '--speed-time', '90', '--speed-limit', '1048576', '--continue-at', '-',
                            '--output', str(partial), row['url'] + '?download=true'],
                           check=True, timeout=2400)
            if partial.stat().st_size != row['bytes'] or sha(partial) != row['sha256']:
                raise RuntimeError('Downloaded file failed size/SHA256: ' + row['name'])
            partial.replace(path)
        elif path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError('Existing file failed size/SHA256: ' + row['name'])
        receipt = dict(name=row['name'], bytes=row['bytes'], sha256=row['sha256'],
                       verified=True, seconds=time.monotonic() - began)
        print('verified', json.dumps(receipt), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, manifest['files']))
    report = dict(checkpoint=manifest['checkpoint'], manifest_sha256=sha(args.manifest),
                  files=results, total_bytes=sum(row['bytes'] for row in results),
                  all_verified=all(row['verified'] for row in results))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('All selected BF16 table files verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proxy', help='Optional HTTP/SOCKS proxy for nodes without internet DNS')
    main(parser.parse_args())
