#!/usr/bin/env python3
import concurrent.futures,json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parent
raw=root/'raw';captures=raw/'captures'
end=time.monotonic()+1800
while not (captures/'rank1/layer47/post_mlp_residual.bin').exists():
 if time.monotonic()>end:raise TimeoutError('capture not complete')
 time.sleep(2)
layers=[captures/f'rank{r}/layer{l}' for r in range(2) for l in range(48)]
layers.sort(key=lambda p:(not (p/'qsa.meta').exists(),str(p)))
def audit(p):
 env=dict(os.environ);env['OMP_NUM_THREADS']='2'
 t=time.monotonic();r=subprocess.run([str(raw/'audit'),str(p)],capture_output=True,text=True,env=env,timeout=600)
 if r.returncode:raise RuntimeError(f'{p}: {r.stderr}; {r.stdout}')
 x=json.loads(r.stdout);x['elapsed_seconds']=time.monotonic()-t
 return x
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 tasks={pool.submit(audit,p):p for p in layers}
 for f in concurrent.futures.as_completed(tasks):
  x=f.result();results.append(x);(raw/'operator-audit.json').write_text(json.dumps(results,indent=2)+'\n')
  print(len(results),x['path'],x['kind'],'done',flush=True)
print('All 96 layer captures audited',flush=True)
