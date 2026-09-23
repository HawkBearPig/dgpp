#!/usr/bin/env python3
"""Verify the actual FP32-result to BF16-cache boundary between first two calls."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('captures',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for rank in (0,1):
 root=a.captures/f'rank{rank}'/'request1'
 if not root.exists():continue
 for layer in range(3):
  old=root/'position0'/f'layer{layer}_gdn_final_state';new=root/'position2048'/f'layer{layer}_gdn_initial_state'
  before=json.loads(old.with_suffix('.json').read_text());after=json.loads(new.with_suffix('.json').read_text())
  assert before['dtype']=='torch.float32' and after['dtype']=='torch.bfloat16'
  x=np.fromfile(old.with_suffix('.bin'),dtype='<f4');bits=x.view(np.uint32)
  expected=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
  actual=np.fromfile(new.with_suffix('.bin'),dtype='<u2');y=(actual.astype(np.uint32)<<16).view(np.float32)
  row=dict(rank=rank,layer=layer,first_output_dtype=before['dtype'],next_input_dtype=after['dtype'],elements=int(x.size),
           matches_bf16_rounding=bool(np.array_equal(actual,expected)),changed_values=int(np.count_nonzero(x!=y)),
           relative_l2=float(np.linalg.norm(x.astype(np.float64)-y)/np.linalg.norm(x.astype(np.float64))),
           max_absolute=float(np.max(np.abs(x-y))))
  assert row['matches_bf16_rounding'];rows.append(row)
a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
