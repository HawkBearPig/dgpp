#!/usr/bin/env python3
"""Serial exact-fixture capture with production restoration in finally."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path('/home/stephen/workspace/dgpp')
WORK = Path('/tmp/dgpp-issue4-gdn-fix-20260923')
RECORD = Path(__file__).resolve().parent
RAW = RECORD / 'raw/replay-campaign'
PROD = ROOT / 'deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json'
BINARY = RECORD / 'raw/gdn-chunk-dgpp-serve'
TEST_ENV = dict(os.environ, DGPP_ENV_FILE=str(Path('/tmp/dgpp-issue4-20260923/benchmarks/results/2026-09-23-issue4-exact/raw/site.env')))
PROD_ENV = dict(os.environ, DGPP_ENV_FILE=str(ROOT / '.env'))
EXPECTED_PROD = '4638e3c595a2765d6d50c4626a9fef3aa2aa82936c6c50e3d0eec871fcde8e00'
receipts = []


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(name, command, env=PROD_ENV, cwd=ROOT, timeout=1200):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), name, 'start', flush=True)
    with (RAW / (name + '.log')).open('w') as f:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    receipts.append({'name': name, 'command': command, 'returncode': result.returncode})
    (RAW / 'commands.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(name + ' failed: ' + (RAW / (name + '.log')).read_text()[-2500:])


def metrics(port):
    return json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=10))


def production_identity():
    config = json.loads((RAW / 'production-resolved.json').read_text())
    command = 'pid=$(pgrep -x dgpp-serve); test "$(printf "%s\\n" "$pid" | wc -l)" -eq 1 && sha256sum /proc/$pid/exe'
    result = {}
    for node in config['nodes']:
        call = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'stephen@' + node, command], capture_output=True, text=True, timeout=20)
        assert call.returncode == 0 and call.stdout.split()[0] == EXPECTED_PROD, (node, call.stdout, call.stderr)
        result[node] = call.stdout
    return result


assert sha(ROOT / 'build-release/dgpp-serve') == EXPECTED_PROD
before = metrics(18080)
assert before['scheduler']['active'] == before['scheduler']['queued'] == 0
assert not before['service']['engine_failed']
(RAW / 'production-before.json').write_text(json.dumps({'metrics': before, 'binaries': production_identity()}, indent=2))
for rank in (0,1):
    capture=RECORD/f'raw/captures/rank{rank}/request1/position0'
    assert len(list(capture.glob('marlin-bundle*.pt')))==1, 'Need one complete exported operator bundle per rank'
    assert (capture/'layer0_moe_routed_sum.json').exists()
    assert (capture/'layer0_moe_expert_down.json').exists()
stopped = active = False
try:
    stopped = True
    run('production-down', [sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'down', '--config', str(PROD)], timeout=120)
    run('isolated-marlin-replays',[sys.executable,str(RECORD/'run_replays.py')],TEST_ENV,WORK,1500)

finally:
    try:
        if active:
            run('capture-down', [sys.executable, str(WORK / 'scripts/dgpp-cluster'), 'down', '--config', str(active_config), '--log-dir', str(active_log)], TEST_ENV, WORK, 120)
    finally:
        if stopped:
            run('production-restore', [sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'up', '--config', str(PROD), '--bin', str(ROOT / 'build-release/dgpp-serve')])
            binaries = production_identity()
            resolved = json.loads(Path('/home/stephen/dgpp/log/deployments/a911a2d7c4b3be6a/cluster.resolved.json').read_text())
            assert resolved == json.loads((RAW / 'production-resolved.json').read_text())
            request = {'model': resolved['model'], 'messages': [{'role': 'user', 'content': 'Reply exactly OK.'}], 'temperature': 0, 'max_tokens': 128, 'reasoning_effort': 'minimal'}
            response = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions', data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'}), timeout=90))
            assert response['choices'][0]['message']['content'].strip() == 'OK'
            m = metrics(18080)
            assert m['scheduler']['active'] == m['scheduler']['queued'] == 0 and not m['service']['engine_failed']
            (RAW / 'restoration.json').write_text(json.dumps({'binaries': binaries, 'config_matches': True, 'smoke': response, 'idle': True}, indent=2))
            print('Production restored and independently verified', flush=True)

print('Isolated Marlin replay campaign complete', flush=True)
