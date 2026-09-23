#!/usr/bin/env python3
"""Publish small result receipts after the FP8 campaign has restored production."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'


def read(path):
    return json.loads(path.read_text())


restore = read(RAW / 'restoration.json')
assert restore['config_matches'] and restore['idle']
weights = read(RAW / 'weight-checks.json')
assert set(weights) == {'0', '1'}
assert all(row['all_pass'] and len(row['checks']) == 24 for row in weights.values())
responses = [read(RAW / f'response-{i}.json') for i in (1, 2)]
verdicts = [read(RAW / f'verdict-{i}.json') for i in (1, 2)]
assert all(row['native_tokens_exact'] and row['protocol_pass'] for row in verdicts)
native = read(ROOT.parent / '2026-09-23-issue4-fp8-control/raw/campaign/native-run/response.json')
nvfp4 = read(ROOT.parent / '2026-09-23-issue4-reference-state-fp32/raw/response-1.json')
native_answer = read(ROOT.parent / '2026-09-23-issue4-fp8-control/verdict.json')['actual']
nvfp4_answer = read(ROOT.parent / '2026-09-23-issue4-reference-state-fp32/verdict-1.json')['actual']


def text(response):
    return response['choices'][0]['message']['content']


rows = []
for request, response in enumerate(responses, 1):
    rows.append(dict(request=request,
                     assistant_text_sha256=hashlib.sha256(text(response).encode()).hexdigest(),
                     same_text_as_native_fp8=text(response) == text(native),
                     same_text_as_nvfp4_reference=text(response) == text(nvfp4),
                     same_parsed_answer_as_native_fp8=verdicts[request - 1]['actual'] == native_answer,
                     same_parsed_answer_as_nvfp4_reference=verdicts[request - 1]['actual'] == nvfp4_answer,
                     completion_tokens=response['usage']['completion_tokens'],
                     correct=verdicts[request - 1]['correct'],
                     failed_key_value=verdicts[request - 1]['actual']['key_0769e0226c63']))
(ROOT / 'response-parity.json').write_text(json.dumps(rows, indent=2) + '\n')
for name in ('verdict-1.json', 'verdict-2.json', 'repeatability.json', 'restoration.json',
             'runtime-source.json', 'weight-checks.json'):
    shutil.copy2(RAW / name, ROOT / name)
print(json.dumps(rows, indent=2))
