#!/usr/bin/env python3
"""Serial native FP8 TP2 table-precision comparison with exact production restoration."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request

ROOT = Path('/home/stephen/workspace/dgpp')
WORK = Path('/tmp/dgpp-issue4-ple-precision-20260923')
RECORD = Path(__file__).resolve().parent
RAW = RECORD / 'raw/campaign'
PROD = ROOT / 'deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json'
BINARY = WORK / 'build-release/dgpp-serve'
MANIFEST = Path('/home/stephen/.cache/dgpp/issue4-bf16-ple-de4b8e4d/manifest.json')
KNOB = 'DGPP_ISSUE4_PLE_BF16_MANIFEST'
PROD_ENV = dict(os.environ, DGPP_ENV_FILE=str(ROOT / '.env'))
TEST_ENV = dict(os.environ, DGPP_ENV_FILE='/tmp/dgpp-issue4-20260923/benchmarks/results/2026-09-23-issue4-exact/raw/site.env')
TEST_ENV.pop(KNOB, None)
TEST_ENV['DGPP_RESIDENT_CACHE'] = 'off'
PROD_ENV.pop(KNOB, None)
EXPECTED_PROD = '4638e3c595a2765d6d50c4626a9fef3aa2aa82936c6c50e3d0eec871fcde8e00'
receipts = []


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(name, command, env=PROD_ENV, cwd=ROOT, timeout=1200):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), name, 'start', flush=True)
    with (RAW / (name + '.log')).open('w') as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    receipts.append(dict(name=name, command=command, returncode=result.returncode))
    (RAW / 'commands.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(name + ' failed: ' + (RAW / (name + '.log')).read_text()[-2500:])


def metrics():
    return json.load(urllib.request.urlopen('http://127.0.0.1:18080/metrics', timeout=10))


def production_identity():
    config = json.loads((RAW / 'production-resolved.json').read_text())
    command = 'pid=$(pgrep -x dgpp-serve); test "$(printf "%s\\n" "$pid" | wc -l)" -eq 1 && sha256sum /proc/$pid/exe'
    result = {}
    for node in config['nodes']:
        call = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'stephen@' + node, command],
                              capture_output=True, text=True, timeout=20)
        assert call.returncode == 0 and call.stdout.split()[0] == EXPECTED_PROD, node
        result[node] = call.stdout
    return result


def test_identity(mode, expected_binary):
    code = (
        'import hashlib,json,subprocess; from pathlib import Path; '
        'pids=subprocess.check_output(["pgrep","-x","dgpp-serve"],text=True).split(); '
        'assert len(pids)==1; p=Path("/proc")/pids[0]; '
        'entries=(p/"environ").read_bytes().split(b"\\0"); '
        f'key={KNOB.encode()!r}+b"="; '
        'values=[entry[len(key):].decode() for entry in entries if entry.startswith(key)]; '
        'cache=[entry.split(b"=",1)[1].decode() for entry in entries if entry.startswith(b"DGPP_RESIDENT_CACHE=")]; '
        'f=(p/"exe").open("rb"); checksum=hashlib.file_digest(f,"sha256").hexdigest(); f.close(); '
        'print(json.dumps(dict(pid=int(pids[0]),binary_sha256=checksum,ple_override=values,resident_cache=cache)))'
    )
    rows = []
    for node in ('192.168.88.11', '192.168.88.12'):
        output = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'stephen@' + node,
                                          'python3 -c ' + shlex.quote(code)], text=True, timeout=30)
        row = json.loads(output)
        assert row['binary_sha256'] == expected_binary
        assert row['ple_override'] == ([str(MANIFEST)] if mode.startswith('bf16') else [])
        assert row['resident_cache'] == ['off']
        rows.append(dict(node=node, **row))
    (RAW / (mode + '-runtime.json')).write_text(json.dumps(rows, indent=2) + '\n')


def main():
    build = json.loads((RECORD / 'build-preflight.json').read_text())
    assert sha(BINARY) == build['binary_sha256']
    assert sha(RECORD / 'diagnostic.patch') == build['diagnostic_patch_sha256']
    assert sha(ROOT / 'build-release/dgpp-serve') == EXPECTED_PROD
    assert sha(RAW / 'request.json') == 'd056488990bf36217fb95f5c8c1ff7bea46a94561c63ec8a396f022149464a35'
    assert sha(MANIFEST) == sha(RECORD / 'bf16-table-manifest.json')
    assert json.loads((RECORD / 'table-verification.json').read_text())['all_verified']
    assert json.loads((RECORD / 'rank0-verification.json').read_text())['all_verified']
    peer = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'stephen@192.168.88.12',
                                    'sha256sum ' + str(MANIFEST)], text=True)
    assert peer.split()[0] == sha(MANIFEST)
    for name, expected in build['test_executable_sha256'].items():
        assert sha(WORK / 'build-release' / name) == expected
    gpu = json.loads((RECORD / 'gpu-validation.json').read_text())
    assert gpu['passed'] and gpu['total_tests'] == 11 and gpu['binary_sha256'] == sha(BINARY)
    for check in gpu['checks']:
        assert check['failed'] == 0 and sha(RECORD / (check['name'] + '.log')) == check['log_sha256']
    parity = json.loads((RECORD / 'input-parity.json').read_text())
    assert parity['request_sha256'] == sha(RAW / 'request.json')
    assert parity['original_request_json_equal_except_model']
    before = metrics()
    assert before['scheduler']['active'] == before['scheduler']['queued'] == 0
    assert not before['service']['engine_failed']
    (RAW / 'production-before.json').write_text(json.dumps(dict(metrics=before, binaries=production_identity()), indent=2) + '\n')
    stopped = active = False
    config = RAW / 'baseline.json'
    active_log = None
    try:
        stopped = True
        run('production-down', [sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'down', '--config', str(PROD)], timeout=120)
        modes = ['baseline-1', 'bf16-1', 'baseline-2']
        baseline_response = None
        for mode in modes:
            env = dict(TEST_ENV)
            if mode.startswith('bf16'):
                env[KNOB] = str(MANIFEST)
            active = True
            active_log = RAW / (mode + '-world')
            run(mode + '-up', [sys.executable, str(WORK / 'scripts/dgpp-cluster'), 'up', '--config', str(config),
                              '--log-dir', str(active_log), '--bin', str(BINARY)], env, WORK)
            test_identity(mode, build['binary_sha256'])
            out = RAW / (mode + '-run')
            with (RAW / (mode + '-replay.log')).open('w') as output:
                result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/results/2026-09-23-issue4-exact/replay.py'),
                                         '--request', str(RAW / 'request.json'), '--oracle', str(RAW / 'oracle.json'),
                                         '--out', str(out)], cwd=WORK, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=1900)
            assert result.returncode in (0, 1)
            verdict = json.loads((out / 'verdict.json').read_text())
            response = json.loads((out / 'response.json').read_text())
            print(mode, 'verdict:', verdict, flush=True)
            assert verdict['protocol_pass'] and verdict['first_request_verified']
            if mode == 'baseline-1':
                assert verdict['correct'] == 5 and verdict['actual']['key_0769e0226c63'] == 'val_51bf3e137245e273'
                baseline_response = response
            elif mode == 'baseline-2':
                assert response['choices'][0]['message']['content'] == baseline_response['choices'][0]['message']['content']
                assert response['usage'] == baseline_response['usage']
            run(mode + '-down', [sys.executable, str(WORK / 'scripts/dgpp-cluster'), 'down', '--config', str(config),
                                '--log-dir', str(active_log)], env, WORK, 120)
            active = False
        (RAW / 'comparison-complete.json').write_text(json.dumps(dict(completed=True, modes=modes), indent=2) + '\n')
    finally:
        try:
            if active:
                run('test-down', [sys.executable, str(WORK / 'scripts/dgpp-cluster'), 'down', '--config', str(config),
                                  '--log-dir', str(active_log)], TEST_ENV, WORK, 120)
        finally:
            if stopped:
                run('production-restore', [sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'up', '--config', str(PROD),
                                           '--bin', str(ROOT / 'build-release/dgpp-serve')])
                binaries = production_identity()
                resolved = json.loads(Path('/home/stephen/dgpp/log/deployments/a911a2d7c4b3be6a/cluster.resolved.json').read_text())
                assert resolved == json.loads((RAW / 'production-resolved.json').read_text())
                request = dict(model=resolved['model'], messages=[dict(role='user', content='Reply exactly OK.')],
                               temperature=0, max_tokens=128, reasoning_effort='minimal')
                response = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',
                                     data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'}), timeout=90))
                assert response['choices'][0]['message']['content'].strip() == 'OK'
                current = metrics()
                assert current['scheduler']['active'] == current['scheduler']['queued'] == 0 and not current['service']['engine_failed']
                (RAW / 'restoration.json').write_text(json.dumps(dict(binaries=binaries, config_matches=True, smoke=response, idle=True), indent=2) + '\n')
                print('Production restored and independently verified', flush=True)


if __name__ == '__main__':
    main()
