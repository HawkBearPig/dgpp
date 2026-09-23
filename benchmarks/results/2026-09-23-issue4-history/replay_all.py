#!/usr/bin/env python3
import concurrent.futures
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, choices=(0, 1))
args = parser.parse_args()
tasks = []
for rank in (range(2) if args.rank is None else [args.rank]):
    for layer in range(48):
        p = ROOT / f'raw/captures/rank{rank}/layer{layer}'
        tasks.append(('index' if layer % 4 == 3 else 'gdn', p))
    tasks.append(('ple', ROOT / f'raw/captures/rank{rank}/layer1'))
tasks.sort(key=lambda t: (t[0] == 'gdn', t[0], str(t[1])))


def replay(task):
    kind, p = task
    started = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='2' if kind == 'ple' else '1')
    result = subprocess.run([str(ROOT / 'raw/history_replay'), kind, str(p)], capture_output=True, text=True, env=env, timeout=1800)
    if result.returncode:
        raise RuntimeError(f'{kind} {p}: {result.stderr}; {result.stdout}')
    data = json.loads(result.stdout)
    data['elapsed_seconds'] = time.monotonic() - started
    return data


completed = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    jobs = {pool.submit(replay, task): task for task in tasks}
    for f in concurrent.futures.as_completed(jobs):
        result = f.result()
        completed.append(result)
        name = 'history-replay' + ('' if args.rank is None else f'-rank{args.rank}') + '.json'
        (ROOT / 'raw' / name).write_text(json.dumps(completed, indent=2) + '\n')
        print(len(completed), result['kind'], result['path'], 'done', flush=True)
print(f'All {len(tasks)} full-history replays complete', flush=True)
