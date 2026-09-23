#!/usr/bin/env python3
"""Run on the idle third node; always remove the isolated GPU container."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
NODE = 'stephen@192.168.88.13'
REMOTE = '/tmp/dgpp-issue4-router-ties-20260923'
NAME = 'dgpp-issue4-router-ties-20260923'
IMAGE = 'sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a'


def call(args, **kwargs):
    return subprocess.run(['ssh','-o','BatchMode=yes',NODE,shlex.join(args)],check=True,**kwargs)


idle = subprocess.check_output(['ssh','-o','BatchMode=yes',NODE,
    'test -z "$(pgrep -x dgpp-serve)" && test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"'],text=True,timeout=20)
identity = subprocess.check_output(['ssh','-o','BatchMode=yes',NODE,
    shlex.join(['docker','image','inspect','--format','{{.Id}}',IMAGE])],text=True,timeout=20).strip()
assert identity == IMAGE
existing = subprocess.check_output(['ssh','-o','BatchMode=yes',NODE,
    shlex.join(['docker','ps','-aq','--filter','name=^/'+NAME+'$'])],text=True,timeout=20)
assert not existing.strip(), 'Isolated test container already exists'
call(['mkdir','-p',REMOTE+'/input',REMOTE+'/output'],timeout=20)
subprocess.run(['scp','-r',str(ROOT/'raw/fixtures'),str(ROOT/'fixtures.json'),str(ROOT/'compare.py'),NODE+':'+REMOTE+'/input/'],check=True,timeout=120)
code_hash=hashlib.sha256((ROOT/'compare.py').read_bytes()).hexdigest()
start=time.monotonic()
try:
    with (ROOT/'raw/gpu-comparison.log').open('w') as log:
        call(['docker','run','--rm','--gpus','all','--name',NAME,'--network','none',
              '--memory','8g','--memory-swap','8g','-v',REMOTE+'/input:/inputs:ro',
              '-v',REMOTE+'/output:/output','--entrypoint','python3',IMAGE,'/inputs/compare.py'],
             stdout=log,stderr=subprocess.STDOUT,timeout=300)
    subprocess.run(['scp',NODE+':'+REMOTE+'/output/comparison.json',str(ROOT/'comparison.json')],check=True,timeout=60)
    (ROOT/'receipt.json').write_text(json.dumps({'node':NODE,'image':IMAGE,'script_sha256':code_hash,
        'idle_verified':True,'wall_seconds':time.monotonic()-start,'returncode':0},indent=2)+'\n')
finally:
    # Also handles timeout: terminating the Docker client alone is insufficient.
    subprocess.run(['ssh','-o','BatchMode=yes',NODE,shlex.join(['docker','rm','-f',NAME])],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=40)
    remaining = subprocess.check_output(['ssh','-o','BatchMode=yes',NODE,
        shlex.join(['docker','ps','-aq','--filter','name=^/'+NAME+'$'])],text=True,timeout=20)
    assert not remaining.strip(), 'Isolated GPU container cleanup failed'
