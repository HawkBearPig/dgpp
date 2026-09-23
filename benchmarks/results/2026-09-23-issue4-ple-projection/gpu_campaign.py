#!/usr/bin/env python3
"""Run the bounded GPU diagnostic serially and restore the original production."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import urllib.request


ROOT = Path('/home/stephen/workspace/dgpp')
RECORD = Path(__file__).resolve().parent
RAW = RECORD / 'raw/gpu-campaign'
RAW.mkdir(exist_ok=False)
source = ROOT / 'benchmarks/results/2026-09-23-issue4-ple-quantization/campaign.py'
spec = importlib.util.spec_from_file_location('restoration_helpers', source)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.RAW = RAW
shutil.copy2('/home/stephen/dgpp/log/deployments/a911a2d7c4b3be6a/cluster.resolved.json', RAW / 'production-resolved.json')
inputs = json.loads((RECORD / 'gpu-input.json').read_text())
binary = RECORD / 'raw/gpu_probe'
assert helper.sha(binary) == inputs['probe_sha256']
for name, checksum in inputs['input_sha256'].items():
    assert helper.sha(RECORD / 'raw/gpu-input' / name) == checksum
before = helper.metrics()
assert before['scheduler']['active'] == before['scheduler']['queued'] == 0
assert not before['service']['engine_failed']
(RAW / 'production-before.json').write_text(json.dumps(dict(metrics=before, binaries=helper.production_identity()), indent=2)+'\n')
stopped = False
try:
    stopped = True
    helper.run('production-down', [sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'down', '--config', str(helper.PROD)], timeout=120)
    for mode in ('bf16', 'fp32'):
        helper.run(mode, [str(binary), str(RECORD / 'raw/gpu-input'), str(RAW / mode), mode], timeout=180)
finally:
    if stopped:
        helper.run('production-restore', [sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'up', '--config', str(helper.PROD), '--bin', str(ROOT / 'build-release/dgpp-serve')])
        binaries = helper.production_identity()
        resolved = json.loads(Path('/home/stephen/dgpp/log/deployments/a911a2d7c4b3be6a/cluster.resolved.json').read_text())
        assert resolved == json.loads((RAW / 'production-resolved.json').read_text())
        request = dict(model=resolved['model'], messages=[dict(role='user', content='Reply exactly OK.')], temperature=0, max_tokens=128, reasoning_effort='minimal')
        response = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions', data=json.dumps(request).encode(), headers={'Content-Type':'application/json'}), timeout=90))
        assert response['choices'][0]['message']['content'].strip() == 'OK'
        current = helper.metrics()
        assert current['scheduler']['active'] == current['scheduler']['queued'] == 0 and not current['service']['engine_failed']
        (RECORD / 'gpu-restoration.json').write_text(json.dumps(dict(binaries=binaries, config_matches=True, smoke=response, idle=True),indent=2)+'\n')
        print('Production restored and verified',flush=True)
