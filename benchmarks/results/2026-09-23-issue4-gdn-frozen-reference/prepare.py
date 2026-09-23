from pathlib import Path
import numpy as np,json,hashlib
root=Path('/home/stephen/workspace/dgpp/benchmarks/results')
r=root/'2026-09-23-issue4-gdn-frozen-reference';(r/'raw/fixtures').mkdir(parents=True,exist_ok=True)
(r/'.gitignore').write_text('raw/\n__pycache__/\n')
cases=[]
for rank in (0,1):
 for layer in (0,16,32,46):
  p=root/f'2026-09-23-issue4-history/raw/captures/rank{rank}/layer{layer}'
  _,K,V,CW=map(int,(p/'sample.meta').read_text().split())
  N=(p/'sample_a.bin').stat().st_size//2
  qkv=np.memmap(p/'sample_qkv.bin',mode='r',dtype='<u2',shape=(N,2*K+V))
  a=np.memmap(p/'sample_a.bin',mode='r',dtype='<u2',shape=(N,1));b=np.memmap(p/'sample_b.bin',mode='r',dtype='<u2',shape=(N,1))
  core=np.memmap(p/'sample_core.bin',mode='r',dtype='<u2',shape=(N,V))
  states=np.memmap(p/'sample_states.bin',mode='r',dtype='<f4',shape=(128,V,K))
  for pos,count in ((0,2048),(260096,1194)):
   out=r/f'raw/fixtures/rank{rank}-layer{layer}-position{pos}.npz'
   np.savez(out,qkv=qkv[pos:pos+count],a=a[pos:pos+count],b=b[pos:pos+count],
    alog=np.fromfile(p/'sample_alog.bin',dtype='<f4'),dt=np.fromfile(p/'sample_dt.bin',dtype='<f4'),
    core=core[pos:pos+count],state_before=np.zeros((1,1,V,K),np.float32) if pos==0 else states[pos//2048-1].copy().reshape(1,1,V,K),
    state_after=states[pos//2048].copy().reshape(1,1,V,K),scale=np.fromfile(p/'sample_scale.bin',dtype='<f4'))
   cases.append(dict(rank=rank,layer=layer,position=pos,tokens=count,file=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
(r/'fixtures.json').write_text(json.dumps(cases,indent=2)+'\n');print(len(cases),'cases',sum(p.stat().st_size for p in (r/'raw/fixtures').iterdir()),'bytes')
