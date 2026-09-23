#!/usr/bin/env python3
"""Verify retained tensor bytes against their independently written trace metadata."""
import argparse,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('captures',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for rank in sorted(a.captures.glob('rank*')):
 for request in sorted(rank.glob('request*')):
  row=dict(rank=rank.name,request=request.name,fields=0,payload_bytes=0,calls=0,orphan_payloads=[])
  for position in sorted(request.glob('position*'),key=lambda x:int(x.name[8:])):
   row['calls']+=1
   metadata={x.stem:x for x in position.glob('layer*.json')}
   payloads={x.stem:x for x in position.glob('layer*.bin')}
   assert set(metadata)<=set(payloads), (position,'missing payloads',set(metadata)-set(payloads))
   row['orphan_payloads'] += [str(payloads[x].relative_to(a.captures)) for x in set(payloads)-set(metadata)]
   for stem,path in metadata.items():
    m=json.loads(path.read_text());data=payloads[stem].read_bytes()
    assert len(data)==m['bytes'],path
    itemsize={'torch.bfloat16':2,'torch.float32':4,'torch.float64':8,'torch.int32':4,'torch.int64':8}[m['dtype']]
    assert len(data)==math.prod(m['saved_shape'])*itemsize,path
    assert hashlib.sha256(data).hexdigest()==m['sha256'],path
    row['fields']+=1;row['payload_bytes']+=len(data)
  assert not row['orphan_payloads'],row
  rows.append(row);print(json.dumps(row),flush=True)
a.output.write_text(json.dumps(rows,indent=2)+'\n')
