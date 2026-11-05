#!/usr/bin/env python3
"""Check the numerical probe's conditional scoring against its full-token scores."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from fabric_logprob import load_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--app',required=True)
    p.add_argument('--checkpoint-dir',required=True)
    args = p.parse_args()
    ids = list(range(1,21))
    command = [args.app,'--checkpoint-dir',args.checkpoint_dir,'--ids',','.join(map(str,ids))]
    with tempfile.TemporaryDirectory(prefix='minimax-scores-') as temporary:
        records = []
        for first in (1,7):
            run = subprocess.run(command+['--score-from',str(first)],capture_output=True,text=True,timeout=120,check=True)
            folder = Path(temporary)/str(first)
            folder.mkdir()
            (folder/'r0.log').write_text(run.stdout)
            steps,ranks = load_run(str(folder),prefill=True)
            if ranks != [0] or [s['target'] for s in steps.values()] != ids[first:]:
                raise AssertionError('conditional probe scored the wrong target positions')
            records.append(steps)
        for step,record in records[1].items():
            if record != records[0][step+6]:
                raise AssertionError('excluding prompt scores changed the conditional logits')
    for bad in ('-1','0','20','21'):
        run = subprocess.run(command+['--score-from',bad],capture_output=True,text=True,timeout=30)
        if run.returncode == 0 or '--score-from' not in run.stderr:
            raise AssertionError(f'invalid score start {bad} was not refused')
    print('MiniMax conditional scoring: exact full-score suffix, correct targets, invalid ranges refused')


if __name__=='__main__':
    main()
