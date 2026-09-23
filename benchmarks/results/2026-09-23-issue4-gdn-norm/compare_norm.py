#!/usr/bin/env python3
"""Compare GDN norm rounding policies on captured real operator inputs."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CAPTURE=ROOT.parent/'2026-09-23-issue4-operators/raw/captures'
def bf(bits):return (np.asarray(bits,dtype=np.uint32)<<16).view(np.float32)
def rb(v):
 b=np.asarray(v,dtype=np.float32).view(np.uint32)
 return bf((b+np.uint32(0x7fff)+((b>>16)&1))>>16)
def rel(a,b):return float(np.linalg.norm(a.astype(np.float64)-b)/np.linalg.norm(b))
records=[]
for rank in (0,1):
 for layer in range(48):
  if layer%4==3:continue
  p=CAPTURE/f'rank{rank}/layer{layer}'
  meta=(p/'gdn.meta').read_text().split();T,KH,VH,KD,D,CW=map(int,meta[:6]);eps=float(meta[6])
  indices=[0,T//2,T-1]
  def got(name):return bf(np.fromfile(p/f'{name}.bin',dtype='<u2').reshape(T,VH,D)[indices]).astype(np.float64)
  x,z,actual=got('core'),got('z'),got('normed')
  w=bf(np.fromfile(p/'norm_weight.bin',dtype='<u2')).astype(np.float64)
  u=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)
  desired=u*w/(1+np.exp(-z))
  fused=rb(desired)
  staged=rb(rb(rb(u)*w)/(1+np.exp(-z)))
  records.append({'rank':rank,'layer':layer,'elements':actual.size,'policy_changed_elements':int((fused!=staged).sum()),'staged_to_real_l2':rel(staged,actual),'staged_to_fp64_l2':rel(staged,desired),'fused_to_fp64_l2':rel(fused,desired)})
summary={'samples':len(records),'elements':sum(r['elements'] for r in records),'changed_elements':sum(r['policy_changed_elements'] for r in records),'max_staged_to_real_l2':max(r['staged_to_real_l2'] for r in records),'max_staged_to_fp64_l2':max(r['staged_to_fp64_l2'] for r in records),'max_fused_to_fp64_l2':max(r['fused_to_fp64_l2'] for r in records),'fused_lower_error_cases':sum(r['fused_to_fp64_l2']<r['staged_to_fp64_l2'] for r in records)}
(ROOT/'norm-policy-comparison.json').write_text(json.dumps({'summary':summary,'samples':records},indent=2)+'\n');print(summary)
