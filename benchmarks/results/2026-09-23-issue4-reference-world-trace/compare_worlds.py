#!/usr/bin/env python3
"""Compare read-only captures from two fresh worlds up to their token divergence."""
import argparse
import json
from pathlib import Path
import numpy as np

STAGES=['ngram_context','query_start_loc','embedding','attn_in','attn_injection',
        'gdn_q','gdn_k','gdn_v','gdn_g','gdn_beta','gdn_initial_state','gdn_core',
        'gdn_final_state','attn_out','mlp_in','residual','mlp_injection','mlp_out','final_hidden']

def ordering(p):
    layer,stage=p.stem.split('_',1)
    return int(layer[5:]),STAGES.index(stage)

def values(path,meta):
    if meta['dtype']=='torch.bfloat16':
        return (np.fromfile(path,dtype='<u2').astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    return np.fromfile(path,dtype={'torch.float32':'<f4','torch.float64':'<f8','torch.int64':'<i8','torch.int32':'<i4'}[meta['dtype']]).astype(np.float64)

def stats(left,right,a,b):
    x,y=values(left.with_suffix('.bin'),a),values(right.with_suffix('.bin'),b)
    indices=np.flatnonzero(x!=y)
    return dict(position=int(left.parent.name[8:]),field=left.stem,shape=a['saved_shape'],
                different_elements=int(indices.size),elements=int(x.size),
                nonfinite=int((~np.isfinite(x)|~np.isfinite(y)).sum()),
                relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30)),
                max_absolute=float(np.max(np.abs(x-y))),first_indices=indices[:16].tolist(),
                first_values=x[indices[:16]].tolist(),second_values=y[indices[:16]].tolist())

def compare(first,second,rank):
    first=first/f'rank{rank}'/'request1';second=second/f'rank{rank}'/'request1'
    result=dict(rank=rank,equal_fields=0,different_fields=0,first_difference=None,
                first_input_difference=None,incomplete_tail=None,early_differences=[],kernel_config_comparisons=[])
    for left in sorted(first.glob('position*'),key=lambda p:int(p.name[8:])):
        right=second/left.name
        if not right.exists() or json.loads((left/'input.json').read_text())!=json.loads((right/'input.json').read_text()):
            result['first_input_difference']=dict(position=int(left.name[8:]),reason='missing call or token/position drift');break
        left_fields={p.name for p in left.glob('layer*.json')}
        right_fields={p.name for p in right.glob('layer*.json')}
        incomplete = left_fields != right_fields
        if incomplete:
            last_left=max(int(p.name[8:]) for p in first.glob('position*'))
            last_right=max(int(p.name[8:]) for p in second.glob('position*'))
            assert int(left.name[8:])==min(last_left,last_right), 'Missing field before final trace call'
            result['incomplete_tail']=dict(position=int(left.name[8:]),
                missing_in_second=sorted(left_fields-right_fields),missing_in_first=sorted(right_fields-left_fields))
        for p in sorted((left/name for name in left_fields & right_fields),key=ordering):
            other=right/p.name;a=json.loads(p.read_text());b=json.loads(other.read_text())
            assert all(a[k]==b[k] for k in ('dtype','saved_shape','original_shape')),(p,other)
            if a['sha256']==b['sha256']:result['equal_fields']+=1;continue
            result['different_fields']+=1
            if result['first_difference'] is None:result['first_difference']=stats(p,other,a,b)
            if int(left.name[8:]) in (0,2048) and int(p.stem.split('_')[0][5:])<4:
                result['early_differences'].append(stats(p,other,a,b))
        config=left/'gdn-kernel-configs.json'
        if config.exists():
            a=json.loads(config.read_text());b=json.loads((right/config.name).read_text())
            result['kernel_config_comparisons'].append(dict(position=int(left.name[8:]),equal=a==b,
                different={k:dict(first=a[k],second=b[k]) for k in a if a[k]!=b[k]}))
        if incomplete:
            break
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=[compare(a.first,a.second,rank) for rank in (0,1)]
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='early_differences'} for row in out],indent=2))
