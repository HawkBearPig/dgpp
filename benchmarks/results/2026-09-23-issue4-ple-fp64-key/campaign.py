#!/usr/bin/env python3
"""Serial native TP2 PLE key-accumulation comparison with exact production restoration."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/home/stephen/workspace/dgpp')
WORK = Path('/tmp/dgpp-issue4-ple-fp64-key-20260923')
RECORD = Path(__file__).resolve().parent
RAW = RECORD / 'raw/campaign'
PROD = ROOT / 'deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json'
BINARY = WORK / 'build-release/dgpp-serve'
SOURCE = ROOT / 'benchmarks/results/2026-09-23-issue4-ple-projection'
KNOB = 'DGPP_ISSUE4_PLE_FP64_KEY'
PROD_ENV = dict(os.environ, DGPP_ENV_FILE=str(ROOT / '.env'))
TEST_ENV = dict(os.environ, DGPP_ENV_FILE='/tmp/dgpp-issue4-20260923/benchmarks/results/2026-09-23-issue4-exact/raw/site.env')
for env in (TEST_ENV, PROD_ENV):
    env.pop(KNOB, None)
    env.pop('DGPP_ISSUE4_PLE_BF16_MANIFEST', None)
EXPECTED_PROD = '4638e3c595a2765d6d50c4626a9fef3aa2aa82936c6c50e3d0eec871fcde8e00'
receipts = []


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(name, command, env=PROD_ENV, cwd=ROOT, timeout=1200):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), name, 'start', flush=True)
    start = time.monotonic()
    with (RAW / (name + '.log')).open('w') as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    receipts.append(dict(name=name, command=command, returncode=result.returncode, wall_seconds=time.monotonic()-start))
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
        'f=(p/"exe").open("rb"); checksum=hashlib.file_digest(f,"sha256").hexdigest(); f.close(); '
        'print(json.dumps(dict(pid=int(pids[0]),binary_sha256=checksum,ple_fp64_key=values)))'
    )
    rows = []
    for node in ('192.168.88.11', '192.168.88.12'):
        output = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'stephen@' + node,
                                          'python3 -c ' + shlex.quote(code)], text=True, timeout=30)
        row = json.loads(output)
        assert row['binary_sha256'] == expected_binary
        assert row['ple_fp64_key'] == (['1'] if mode.startswith('fp64') else [])
        rows.append(dict(node=node, **row))
    (RAW / (mode + '-runtime.json')).write_text(json.dumps(rows, indent=2) + '\n')


def main():
    build = json.loads((RECORD / 'build-preflight.json').read_text())
    assert sha(BINARY) == build['binary_sha256']
    assert sha(RECORD / 'diagnostic.patch') == build['diagnostic_patch_sha256']
    assert sha(ROOT / 'build-release/dgpp-serve') == EXPECTED_PROD
    assert sha(RAW / 'request.json') == '9ebc3a1ed20f37f3501bf4629bf84d3e0523cbb33989c3ce68fb5e8fd0db2393'
    assert sha(RECORD / 'raw/gpu_probe') == build['probe_sha256']
    inputs = json.loads((SOURCE / 'gpu-input.json').read_text())
    for name, expected in inputs['input_sha256'].items():
        assert sha(SOURCE / 'raw/gpu-input' / name) == expected
    assert sha(RAW / 'oracle.json') == 'fb8dde5737c1a9bf78e89188f2fd8ba09bd7737faa51f6795458813d3ceafc64'
    assert json.loads((RAW / 'baseline.json').read_text()) == json.loads((ROOT / 'benchmarks/results/2026-09-23-issue4-ple-quantization/raw/campaign/baseline.json').read_text())
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
        for mode, rows in [('bf16', 2048), ('fp64', 2048), ('fp64-tail', 1024)]:
            run('gpu-' + mode, [str(RECORD / 'raw/gpu_probe'), str(SOURCE / 'raw/gpu-input'),
                               str(RAW / ('gpu-' + mode)), 'bf16' if mode == 'bf16' else 'fp64', str(rows)],
                TEST_ENV, WORK, 180)
        run('gpu-validation', [str(ROOT / '.venv/bin/python3'), str(RECORD / 'analyze_gpu.py')],
            dict(TEST_ENV, OPENBLAS_NUM_THREADS='4'), WORK, 180)
        assert json.loads((RECORD / 'gpu-validation.json').read_text())['passed']
        modes = ['baseline-1', 'fp64-1', 'baseline-2']
        baseline_response = None
        for mode in modes:
            env = dict(TEST_ENV)
            if mode.startswith('fp64'):
                env[KNOB] = '1'
            active = True
            active_log = RAW / (mode + '-world')
            run(mode + '-up', [sys.executable, str(WORK / 'scripts/dgpp-cluster'), 'up', '--config', str(config),
                              '--log-dir', str(active_log), '--bin', str(BINARY)], env, WORK)
            test_identity(mode, build['binary_sha256'])
            out = RAW / (mode + '-run')
            with (RAW / (mode + '-replay.log')).open('w') as output:
                result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/results/2026-09-23-issue4-exact/replay.py'),
                                         '--request', str(RAW / 'request.json'), '--oracle', str(RAW / 'oracle.json'),
                                         '--out', str(out)], cwd=WORK, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=2400)
            assert result.returncode in (0, 1)
            verdict = json.loads((out / 'verdict.json').read_text())
            response = json.loads((out / 'response.json').read_text())
            print(mode, 'verdict:', verdict, flush=True)
            assert verdict['protocol_pass'] and verdict['first_request_verified']
            if mode == 'baseline-1':
                assert verdict['correct'] == 5 and verdict['actual']['key_0769e0226c63'] == 'val_5dac9ed720abddaf'
                baseline_response = response
            elif mode == 'baseline-2':
                assert response['choices'][0]['message']['content'] == baseline_response['choices'][0]['message']['content']
                assert response['usage'] == baseline_response['usage']
            resolved_files = list(active_log.rglob('cluster.resolved.json'))
            assert len(resolved_files) == 1, resolved_files
            resolved = json.loads(resolved_files[0].read_text())
            assert len(resolved['nodes']) == 2
            resolved['paths'].pop('log_dir', None)
            if mode == 'baseline-1':
                baseline_config = resolved
            else:
                assert resolved == baseline_config
            log_checks = []
            peer_path = resolved['paths']['stage_dir'] + '/serve_r1.log'
            peer_contents = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'stephen@192.168.88.12',
                                                     'cat ' + shlex.quote(peer_path)], timeout=30)
            (active_log / 'serve_r1.log').write_bytes(peer_contents)
            peer_logs = list(active_log.glob('serve_r*.log'))
            assert len(peer_logs) == 2, peer_logs
            for log in peer_logs:
                text = log.read_text(errors='replace').rsplit(' INFO  dgpp-serve ', 1)[-1]
                enabled = 'executing FP64 PLE key projection rows=2048 output=10240 input=1280' in text
                assert enabled == mode.startswith('fp64'), (mode, str(log), enabled)
                log_checks.append(dict(log=str(log), actual_fp64_prefill=enabled))
            (RAW / (mode + '-activation.json')).write_text(json.dumps(log_checks, indent=2)+'\n')
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
