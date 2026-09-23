#!/usr/bin/env python3
"""Verify actual FP32 GDN cache storage between the first two model calls."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('captures',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for rank in (0,1):
 for request in (1,2):
  root=a.captures/f'rank{rank}'/f'request{request}'
  for layer in range(3):
   old=root/'position0'/f'layer{layer}_gdn_final_state';new=root/'position2048'/f'layer{layer}_gdn_initial_state'
   before=json.loads(old.with_suffix('.json').read_text());after=json.loads(new.with_suffix('.json').read_text())
   assert before['dtype']==after['dtype']=='torch.float32',(old,before,after)
   x=np.fromfile(old.with_suffix('.bin'),dtype='<f4');y=np.fromfile(new.with_suffix('.bin'),dtype='<f4')
   assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
   exact=old.with_suffix('.bin').read_bytes()==new.with_suffix('.bin').read_bytes()
   row=dict(rank=rank,request=request,layer=layer,first_output_dtype=before['dtype'],next_input_dtype=after['dtype'],elements=int(x.size),bitwise_exact=exact)
   assert exact,row;rows.append(row)
a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
