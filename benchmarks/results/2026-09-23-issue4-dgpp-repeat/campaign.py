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
import time
import importlib.util
import urllib.request

ROOT = Path('/home/stephen/workspace/dgpp')
WORK = Path('/tmp/dgpp-issue4-20260923')
RECORD = Path(__file__).resolve().parent
RAW = RECORD / 'raw/campaign'
PROD = ROOT / 'deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json'
BINARY = ROOT / 'benchmarks/results/2026-09-23-issue4-operators/raw/clean-dgpp-serve'
TEST_ENV = dict(os.environ, DGPP_ENV_FILE=str(WORK / 'benchmarks/results/2026-09-23-issue4-exact/raw/site.env'))
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
build = {'base_commit': 'c6ca191863d620360f3b4370c0156f9e26a2631f', 'binary_sha256': sha(BINARY), 'request_sha256': sha(RAW / 'request.json')}
(RAW / 'build.json').write_text(json.dumps(build, indent=2))
stopped = active = False
try:
    stopped = True
    run('production-down', [sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'down', '--config', str(PROD)], timeout=120)
    assert sha(BINARY) == 'b6afe5546afb687e4f9adea276625b2cbff2ac38ba6dc60eab55879a50ddba47'
    active = True
    active_config = RAW / 'baseline.json'
    active_log = RAW / 'world'
    run('baseline-up', [sys.executable, str(WORK / 'scripts/dgpp-cluster'), 'up', '--config', str(active_config), '--log-dir', str(active_log), '--bin', str(BINARY)], TEST_ENV, WORK)
    spec=importlib.util.spec_from_file_location('issue4_replay',ROOT/'benchmarks/results/2026-09-23-issue4-exact/replay.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    texts=[]
    for repetition in (1,2):
        before = metrics(18084)
        assert before['scheduler']['active'] == before['scheduler']['queued'] == 0
        assert before['service']['requests_total'] == before['scheduler']['prompts_prefilled'] == repetition-1
        (RAW/f'metrics-before-{repetition}.json').write_text(json.dumps(before,indent=2)+'\n')
        request=urllib.request.Request('http://127.0.0.1:18084/v1/chat/completions',data=(RAW/'request.json').read_bytes(),headers={'Content-Type':'application/json'})
        print('Unchanged DGPP request',repetition,'started',flush=True)
        start=time.monotonic()
        response=json.load(urllib.request.urlopen(request,timeout=1800))
        (RAW/f'response-{repetition}.json').write_text(json.dumps(response,indent=2)+'\n')
        verdict=module.score(response,json.loads((RAW/'oracle.json').read_text()))
        verdict['wall_seconds']=time.monotonic()-start
        assert verdict['protocol_pass']
        after=metrics(18084)
        assert after['service']['requests_total'] == after['scheduler']['prompts_prefilled'] == repetition
        (RAW/f'metrics-after-{repetition}.json').write_text(json.dumps(after,indent=2)+'\n')
        (RAW/f'verdict-{repetition}.json').write_text(json.dumps(verdict,indent=2)+'\n')
        texts.append(response['choices'][0]['message']['content'])
        print('DGPP verdict',repetition,verdict,flush=True)
    (RAW/'repeatability.json').write_text(json.dumps({'same_assistant_text':texts[0]==texts[1]},indent=2)+'\n')
    run('baseline-down', [sys.executable, str(WORK / 'scripts/dgpp-cluster'), 'down', '--config', str(active_config), '--log-dir', str(active_log)], TEST_ENV, WORK,120)
    active=False

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

print('DGPP repeatability diagnostics complete', flush=True)
