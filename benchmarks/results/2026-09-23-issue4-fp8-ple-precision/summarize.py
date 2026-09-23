#!/usr/bin/env python3
"""Rescore and bind the completed native TP2 table-precision comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    score = runpy.run_path(str(args.scorer))['score']
    oracle = json.loads((args.raw / 'oracle.json').read_text())
    restored = json.loads((args.raw / 'restoration.json').read_text())
    complete = json.loads((args.raw / 'comparison-complete.json').read_text())
    modes = ['baseline-1', 'bf16-1', 'baseline-2']
    assert complete == dict(completed=True, modes=modes)
    assert restored['config_matches'] and restored['idle']
    assert restored['smoke']['choices'][0]['message']['content'].strip() == 'OK'
    assert len(restored['binaries']) == 4
    assert all(value.split()[0] == '4638e3c595a2765d6d50c4626a9fef3aa2aa82936c6c50e3d0eec871fcde8e00'
               for value in restored['binaries'].values())
    responses, configs, rows, loader_receipts = {}, {}, [], []
    for mode in modes:
        root = args.raw / (mode + '-run')
        assert sha(root / 'request.json') == 'd056488990bf36217fb95f5c8c1ff7bea46a94561c63ec8a396f022149464a35'
        assert json.loads((root / 'oracle.json').read_text()) == oracle
        response = json.loads((root / 'response.json').read_text())
        responses[mode] = response
        result = score(response, oracle)
        retained = json.loads((root / 'verdict.json').read_text())
        for key, value in result.items():
            assert retained[key] == value, (mode, key)
        assert result['protocol_pass'] and retained['first_request_verified']
        transport = json.loads((root / 'transport.json').read_text())
        assert transport['status'] == 200
        runtime = json.loads((args.raw / (mode + '-runtime.json')).read_text())
        assert len(runtime) == 2
        for rank in runtime:
            assert rank['binary_sha256'] == args.binary_sha256
            expected = ['/home/stephen/.cache/dgpp/issue4-bf16-ple-de4b8e4d/manifest.json'] if mode == 'bf16-1' else []
            assert rank['ple_override'] == expected
            assert rank['resident_cache'] == ['off']
        configs[mode] = json.loads((args.raw / (mode + '-world/cluster.resolved.json')).read_text())
        for rank in (0, 1):
            log = args.raw / (mode + f'-world/serve_r{rank}.log')
            text = log.read_text()
            # The peer deployment appends successive launches to one log.
            # Check only this launch, retaining both full-log and window hashes.
            lines = text.splitlines()
            starts = [i for i, line in enumerate(lines) if ' INFO  dgpp-serve ' in line]
            assert starts, (mode, rank, 'missing launch marker')
            window = '\n'.join(lines[starts[-1]:])
            assert 'resident image cache off (DGPP_RESIDENT_CACHE)' in window
            if mode == 'bf16-1':
                assert f'issue4: rank {rank} uses BF16 PLE rows' in window
            else:
                assert 'uses BF16 PLE rows' not in window
            loader_receipts.append(dict(mode=mode, rank=rank, launches_in_log=len(starts),
                                        launch_start=lines[starts[-1]], log_sha256=sha(log),
                                        current_launch_sha256=hashlib.sha256(window.encode()).hexdigest(),
                                        bf16_rows_enabled='uses BF16 PLE rows' in window))
        rows.append(dict(mode=mode, correct=result['correct'], total=result['total'],
                         failed_key_value=result['actual']['key_0769e0226c63'],
                         completion_tokens=result['usage']['completion_tokens'],
                         wall_seconds=transport['wall_seconds'], pass_original_request=retained['pass']))
        for name in ('verdict.json', 'response.json', 'transport.json'):
            shutil.copy2(root / name, args.output / (mode + '-' + name))
        shutil.copy2(args.raw / (mode + '-runtime.json'), args.output / (mode + '-runtime.json'))
    a, b = responses['baseline-1'], responses['baseline-2']
    assert a['choices'][0]['message']['content'] == b['choices'][0]['message']['content']
    assert a['usage'] == b['usage']
    middle = responses['bf16-1']
    for mode in modes:
        assert len(configs[mode]['nodes']) == 2
        assert configs[mode]['paths'].pop('log_dir') == str(args.raw / (mode + '-world'))
    assert all(configs[mode] == configs['baseline-1'] for mode in modes)
    result = dict(modes=rows, request_bytes_identical_across_modes=True,
                  original_request_json_equal_except_model=True,
                  baseline_full_text_and_usage_repeat=True, effective_configuration_equal=True,
                  actual_binary_equal_on_both_ranks=True, bf16_override_only_in_middle_run=True,
                  bf16_full_text_and_usage_equal_to_baseline=(
                      middle['choices'][0]['message']['content'] == a['choices'][0]['message']['content']
                      and middle['usage'] == a['usage']), loader_receipts=loader_receipts,
                  production_restored_and_verified=True,
                  script_sha256=sha(Path(__file__)), scorer_sha256=sha(args.scorer),
                  interpretation='This changes only n-gram table values in the native FP8 TP2 execution; it does not test unquantized experts.')
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(args.raw / 'restoration.json', args.output / 'restoration.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--scorer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    main(parser.parse_args())
