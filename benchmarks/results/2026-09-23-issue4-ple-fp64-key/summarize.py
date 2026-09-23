#!/usr/bin/env python3
"""Build a compact report only from completed, independently verified receipts."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw/campaign'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


build = read(ROOT / 'build-preflight.json')
assert read(ROOT / 'gpu-validation.json')['passed']
assert read(RAW / 'comparison-complete.json')['completed']
restoration = read(RAW / 'restoration.json')
assert restoration['config_matches'] and restoration['idle']
assert restoration['smoke']['choices'][0]['message']['content'].strip() == 'OK'
assert len(restoration['binaries']) == 4
assert all(row.startswith('4638e3c595a2765d6d50c4626a9fef3aa2aa82936c6c50e3d0eec871fcde8e00')
           for row in restoration['binaries'].values())
modes = []
responses = []
for mode in ['baseline-1', 'fp64-1', 'baseline-2']:
    run = RAW / (mode + '-run')
    transport = read(run / 'transport.json')
    verdict = read(run / 'verdict.json')
    response = read(run / 'response.json')
    assert sha(run / 'request.json') == transport['request_sha256'] == '9ebc3a1ed20f37f3501bf4629bf84d3e0523cbb33989c3ce68fb5e8fd0db2393'
    assert transport['status'] == 200
    assert verdict['protocol_pass'] and verdict['first_request_verified']
    runtime = read(RAW / (mode + '-runtime.json'))
    assert len(runtime) == 2
    assert all(row['binary_sha256'] == build['binary_sha256'] for row in runtime)
    assert all(row['ple_fp64_key'] == (['1'] if mode.startswith('fp64') else []) for row in runtime)
    activation = read(RAW / (mode + '-activation.json'))
    assert len(activation) == 2
    assert all(row['actual_fp64_prefill'] == mode.startswith('fp64') for row in activation)
    metrics = read(run / 'metrics-after.json')
    assert metrics['scheduler']['active'] == metrics['scheduler']['queued'] == 0
    assert not metrics['service']['engine_failed']
    modes.append(dict(mode=mode, correct=verdict['correct'], actual=verdict['actual'],
                      usage=response['usage'], wall_seconds=transport['wall_seconds'],
                      request_sha256=transport['request_sha256'], first_request_verified=True,
                      actual_rank_code_paths_verified=True,
                      response_sha256=sha(run / 'response.json'),
                      content_sha256=hashlib.sha256(response['choices'][0]['message']['content'].encode()).hexdigest(),
                      runtime=runtime))
    responses.append(response)
assert responses[0]['choices'][0]['message']['content'] == responses[2]['choices'][0]['message']['content']
assert responses[0]['usage'] == responses[2]['usage']
report = dict(completed=True, modes=modes, baseline_text_and_usage_repeat_exactly=True,
              fp64_text_matches_baseline=responses[1]['choices'][0]['message']['content'] == responses[0]['choices'][0]['message']['content'],
              fp64_usage_matches_baseline=responses[1]['usage'] == responses[0]['usage'],
              restoration_verified=True, build=build,
              script_sha256=sha(Path(__file__)))
(ROOT / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
shutil.copy2(RAW / 'restoration.json', ROOT / 'restoration.json')
shutil.copy2(RAW / 'commands.json', ROOT / 'commands.json')
print(json.dumps(dict(modes=[{k:v[k] for k in ['mode','correct','wall_seconds','usage']} for v in modes],
                     fp64_text_matches_baseline=report['fp64_text_matches_baseline']), indent=2))
