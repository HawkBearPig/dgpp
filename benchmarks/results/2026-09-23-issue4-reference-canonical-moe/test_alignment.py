"""GPU grouping checked against an independent CPU enumeration, including padding."""
import json
from pathlib import Path
import numpy as np
import torch
from issue4_moe_align import moe_align_block_size

rng=np.random.default_rng(1447);results=[]
cases=[(m,k,e,b,p) for m,k,e in [(1,1,1),(1,10,512),(7,3,7),(64,10,512),(65,3,17),(2048,10,512)] for b in (8,16,32,48,64) for p in (False,True)]
for m,k,e,block,pad in cases:
    raw=np.stack([rng.choice(e,k,replace=False) for _ in range(m)]).astype(np.int32)
    flat=raw.reshape(-1);expected=[];owners=[]
    for expert in range(e):
        ids=np.flatnonzero(flat==expert).tolist()
        if not ids:continue
        ids += [len(flat)]*((-len(ids))%block)
        expected.extend(ids);owners.extend([expert]*(len(ids)//block))
    x=torch.from_numpy(raw).cuda();hashes=[]
    for repetition in range(3):
        ids,experts,total=moe_align_block_size(x,block,e,pad_sorted_ids=pad,ignore_invalid_experts=True)
        got=ids.cpu().numpy();n=int(total.cpu()[0])
        assert n==len(expected) and got[:n].tolist()==expected
        assert experts[:n//block].cpu().tolist()==owners
        assert np.all(got[n:]==len(flat))
        hashes.append(got.tobytes())
    assert hashes[0]==hashes[1]==hashes[2]
    results.append(dict(tokens=m,top_k=k,experts=e,block_size=block,pad_sorted_ids=pad,valid_padded_tokens=len(expected),pass_=True))
Path('/output/alignment-tests.json').write_text(json.dumps({'cases':results,'all_pass':True},indent=2)+'\n')
print('Canonical grouping:',len(results),'CPU-oracle GPU cases passed',flush=True)
