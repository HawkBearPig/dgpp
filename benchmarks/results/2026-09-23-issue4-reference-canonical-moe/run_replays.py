#!/usr/bin/env python3
"""Isolated single-GPU replays; invoke only while the production GPUs are idle."""
from pathlib import Path
import hashlib,json,subprocess,time
R=Path(__file__).resolve().parent
IMAGE='sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a'
NAME='dgpp-issue4-canonical-marlin-replay-20260923'
assert subprocess.run(['pgrep','-x','dgpp-serve'],stdout=subprocess.DEVNULL).returncode==1
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',IMAGE],text=True).strip()==IMAGE
assert not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+NAME+'$'],text=True).strip()
receipt={'image':IMAGE,'script_sha256':hashlib.sha256((R/'replay_experts.py').read_bytes()).hexdigest(),'runs':[]}
try:
 tests=R/'raw/alignment-unit';tests.mkdir(exist_ok=False)
 with (tests/'run.log').open('w') as log:
  test=subprocess.run(['docker','run','--rm','--pull','never','--gpus','all','--name',NAME,
      '--network','none','--memory','8g','--memory-swap','8g','-v',str(R)+':/source:ro',
      '-v',str(tests)+':/output','--entrypoint','python3',IMAGE,'/source/test_alignment.py'],
      stdout=log,stderr=subprocess.STDOUT,timeout=180)
 assert test.returncode==0,(tests/'run.log').read_text()[-2500:]
 assert json.loads((tests/'alignment-tests.json').read_text())['all_pass']
 for rank in (0,1):
  source=R.parent/f'2026-09-23-issue4-reference-marlin-probe/raw/captures/rank{rank}/request1/position0'
  bundles=list(source.glob('marlin-bundle*.pt'))
  assert len(bundles)==1,'Multiple MoE chunks need separate replay accounting before this harness is used'
  with (source/'marlin-bundle.pt').open('rb') as f:bundle_sha=hashlib.file_digest(f,'sha256').hexdigest()
  for process in (1,2):
   out=R/f'raw/replays/rank{rank}-process{process}';out.mkdir(parents=True,exist_ok=False)
   start=time.monotonic()
   with (out/'run.log').open('w') as log:
    call=subprocess.run(['docker','run','--rm','--pull','never','--gpus','all','--name',NAME,
        '--network','none','--memory','32g','--memory-swap','32g',
        '-v',str(source)+':/capture:ro','-v',str(R)+':/source:ro',
        '-v',str(out)+':/output','--entrypoint','python3',IMAGE,'/source/replay_experts.py','/capture','/output'],
        stdout=log,stderr=subprocess.STDOUT,timeout=300)
   row=dict(rank=rank,process=process,returncode=call.returncode,wall_seconds=time.monotonic()-start,bundle_sha256=bundle_sha)
   receipt['runs'].append(row);(R/'replay-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
   assert call.returncode==0,(row,(out/'run.log').read_text()[-3000:])
   result=json.loads((out/'comparison.json').read_text());assert result['complete']
   print(row,flush=True)
finally:
 subprocess.run(['docker','rm','-f',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=40)
 assert not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+NAME+'$'],text=True).strip()
receipt['complete']=True;(R/'replay-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
