#!/usr/bin/env python3
import importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('a',ROOT.parent/'2026-09-23-issue4-moe-audit/audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
results=[]
for rank in (0,1):
 for layer in range(3,48,4):
  p=ROOT.parent/f'2026-09-23-issue4-operators/raw/captures/rank{rank}/layer{layer}'
  meta=(p/'qsa.meta').read_text().split();T,H,_,D=map(int,meta[:4]);eps=float(meta[9]);R=int(meta[10]);last=int(meta[11]);half=R//2
  q=np.fromfile(p/'q.bin',dtype='<u2').reshape(1,H,2*D)
  qn=np.fromfile(p/'qn.bin',dtype='<u2').reshape(1,H,D)
  w=a.bf16(np.fromfile(p/'q_norm.bin',dtype='<u2')).astype(np.float64)
  inv=np.fromfile(p/'inv_freq.bin',dtype='<f4')
  for row in (0,):
   x=a.bf16(q[row,:,:D]).astype(np.float64)
   z=a.rounded(x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*(1+w))
   angle=np.float32(last)*inv
   c=a.rounded(np.cos(angle));s=a.rounded(np.sin(angle))
   left=z[:,:half].astype(np.float64);right=z[:,half:R].astype(np.float64)
   old=z.copy();new=z.copy()
   old[:,:half]=a.rounded(a.rounded(left*c)-a.rounded(right*s))
   old[:,half:R]=a.rounded(a.rounded(right*c)+a.rounded(left*s))
   new[:,:half]=a.rounded(left*c-right*s);new[:,half:R]=a.rounded(right*c+left*s)
   results.append(dict(rank=rank,layer=layer,position=last,changed=int(np.count_nonzero(old!=new)),
      rotary_values=H*R,old_vs_capture=a.stats(old,a.bf16(qn[row])),policies=a.stats(new,old)))
report={'scope':'Independent frozen raw Q replay with FP64 normalization and two explicit rotation storage policies; main Q only, no complete model claim.',
'cases':len(results),'changed_rotary_values':sum(v['changed'] for v in results),'rotary_values':sum(v['rotary_values'] for v in results),
'max_old_vs_capture_l2':max(v['old_vs_capture']['relative_l2'] for v in results),'results':results}
(ROOT/'captured-rotation.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='results'})
