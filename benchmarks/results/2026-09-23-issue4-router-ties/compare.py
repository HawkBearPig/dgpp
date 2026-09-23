#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

root=Path('/inputs')
fixture=json.loads((root/'fixtures.json').read_text())[0]
path=root/'fixtures'/fixture['file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==fixture['sha256']
data=np.load(path)
logits=torch.from_numpy(data['logits']).cuda().to(torch.bfloat16)
expected=torch.from_numpy(data['ids']).cuda()
captured_weights=torch.from_numpy(data['weights']).cuda()
count=logits.shape[0]


def run(x):
    hidden=torch.empty(x.shape[0],2560,dtype=torch.bfloat16,device='cuda')
    weights,ids,_=fused_topk(hidden,x,10,True)
    order=ids.argsort(dim=1)
    return ids.gather(1,order),weights.gather(1,order)


results={}
outputs={}
for batch in (1194,2048):
    x=torch.cat((logits,logits[:1].expand(batch-count,-1)),dim=0).contiguous()
    ids,weights=run(x)
    ids,weights=ids[:count],weights[:count]
    outputs[str(batch)]=(ids.clone(),weights.clone())
singles=[run(logits[i:i+1]) for i in range(count)]
outputs['1']=(torch.cat([x[0] for x in singles]),torch.cat([x[1] for x in singles]))
for name,(ids,weights) in outputs.items():
    changed=(ids!=expected).any(dim=1).nonzero().flatten().cpu().tolist()
    swaps=[]
    for row in changed:
        actual=set(ids[row].cpu().tolist());wanted=set(data['ids'][row].tolist())
        removed,added=sorted(wanted-actual),sorted(actual-wanted)
        values=data['logits'][row]
        threshold=np.sort(values)[-10]
        swaps.append({'metadata':data['metadata'][row].tolist(),'removed':removed,'added':added,
                      'only_cutoff_ties':all(values[j]==threshold for j in removed+added)})
    results[name]={'differing_sets':len(changed),'swaps':swaps,
                  'max_weight_sum_error':(weights.sum(1)-1).abs().max().item(),
                  'bf16_weights_exact':bool(torch.equal(weights.to(torch.bfloat16).float(),captured_weights))}
shape_exact=all(torch.equal(outputs['1194'][0],v[0]) and torch.equal(outputs['1194'][1],v[1]) for v in outputs.values())
result={'rows':count,'cutoff_ties':fixture['cutoff_ties'],'call_shapes_exact':shape_exact,'results':results,
        'selected_sets_match_dgpp':all(x['differing_sets']==0 for x in results.values())}
Path('/output/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
