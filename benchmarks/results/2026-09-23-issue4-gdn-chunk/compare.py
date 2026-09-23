#!/usr/bin/env python3
"""Compare the native block-GDN prototype with the pinned FLA operators."""
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep

ROOT = Path('/inputs')
lib = ctypes.CDLL(str(ROOT/'libgdn_chunk.so'))
lib.issue4_gdn_bytes.argtypes = [ctypes.c_int]*3
lib.issue4_gdn_bytes.restype = ctypes.c_size_t
lib.issue4_gdn.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_int]*3 + [ctypes.c_float,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
lib.issue4_gdn.restype = ctypes.c_int
lib.issue4_gdn_recurrent.argtypes = lib.issue4_gdn.argtypes
lib.issue4_gdn_recurrent.restype = ctypes.c_int


def tensor(a, bf16=False):
    t = torch.from_numpy(a.copy())
    return (t.view(torch.bfloat16) if bf16 else t).cuda()


def stats(r, a):
    r, a = r.double(), a.double()
    assert r.shape == a.shape, (r.shape,a.shape)
    return dict(relative_l2=(torch.linalg.vector_norm(r-a)/torch.linalg.vector_norm(r).clamp_min(1e-30)).item(),
                max_abs=(r-a).abs().max().item(), different=int((r!=a).sum()),
                elements=r.numel(), finite=bool(torch.isfinite(r).all() and torch.isfinite(a).all()))


def native(x,a,b,al,dt,initial,hk,hv,recurrent=False):
    t=x.shape[0]
    n=lib.issue4_gdn_bytes(t,hk,hv)
    ws=torch.empty(n,device='cuda',dtype=torch.uint8)
    state=initial.clone()
    out=torch.empty((1,t,hv,128),device='cuda',dtype=torch.bfloat16)
    args=[p.data_ptr() for p in (x,a,b,al,dt,state,out)]
    fn=lib.issue4_gdn_recurrent if recurrent else lib.issue4_gdn
    assert fn(*args,t,hk,hv,128**-.5,ws.data_ptr(),n,torch.cuda.current_stream().cuda_stream)==0
    torch.cuda.synchronize()
    return out,state,ws


def check(label,x,a,b,al,dt,initial,hk,hv):
    t=x.shape[0]
    q,k,v,g,beta=(p.unsqueeze(0) for p in fused_post_conv_prep(x,a,b,al,dt,hk,128,128,apply_l2norm=True))
    cu=torch.tensor([0,t],device='cuda',dtype=torch.int32)
    out,state=chunk_gated_delta_rule(q,k,v,g,beta,initial_state=initial.clone(),output_final_state=True,
                                   cu_seqlens=cu,use_qk_l2norm_in_kernel=False)
    got,gs,ws=native(x,a,b,al,dt,initial,hk,hv)
    before=time.monotonic()
    repeat,rs,_=native(x,a,b,al,dt,initial,hk,hv)
    elapsed=time.monotonic()-before
    nt=(t+63)//64
    n=nt*hk*64*128*2
    gq=ws[:n].view(torch.bfloat16).reshape(nt,hk,64,128).permute(0,2,1,3).reshape(1,nt*64,hk,128)[:,:t]
    gk=ws[n:2*n].view(torch.bfloat16).reshape(nt,hk,64,128).permute(0,2,1,3).reshape(1,nt*64,hk,128)[:,:t]
    result=dict(label=label,tokens=t,key_heads=hk,value_heads=hv,
                q=stats(q,gq),k=stats(k,gk),output=stats(out,got),state=stats(state,gs),
                repeat_exact=bool(torch.equal(got,repeat) and torch.equal(gs,rs)),
                warm_seconds=elapsed,workspace_bytes=ws.numel())
    result['pass']=(result['repeat_exact'] and all(result[s]['finite'] for s in ('q','k','output','state'))
                    and result['output']['relative_l2']<.002 and result['state']['relative_l2']<.001)
    results.append(result)
    Path('/output/comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return out, state


results=[]
from tuning import pin_baseline
pin_baseline()
fixtures=json.loads((ROOT/'fixtures.json').read_text())
loaded=[]
for case in fixtures:
    p=ROOT/'fixtures'/case['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==case['sha256']
    d=np.load(p)
    x,a,b=(tensor(d[n],True) for n in ('qkv','a','b'))
    al,dt,initial=(tensor(d[n]) for n in ('alog','dt','state_before'))
    check(case['file'],x,a,b,al,dt,initial,1,1)
    if x.shape[0]==1194: loaded.append((x,a,b,al,dt,initial))
    if len(results)==1:
        for t in (1,7,63,64,65,129):
            check(f'boundary-{t}',x[:t].contiguous(),a[:t].contiguous(),b[:t].contiguous(),al,dt,initial,1,1)

# Different saved heads occupy each grouped-key slot; the value/gate/state
# choices rotate independently. Identical replicas would hide indexing bugs.
assert len(loaded)==8
qs=torch.cat([z[0][:,:128] for z in loaded],dim=1)
ks=torch.cat([z[0][:,128:256] for z in loaded],dim=1)
chosen=[loaded[(h+v)%8] for h in range(8) for v in range(3)]
vs=torch.cat([z[0][:,256:] for z in chosen],dim=1)
x=torch.cat((qs,ks,vs),dim=1)
a,b=(torch.cat([z[i] for z in chosen],dim=1) for i in (1,2))
al,dt=(torch.cat([z[i].reshape(-1) for z in chosen]) for i in (3,4))
state=torch.cat([z[5] for z in chosen],dim=1)
check('mixed-grouped-heads',x,a,b,al,dt,state,8,24)
grouped=(x,a,b,al,dt,state,8,24)

# Small deterministic fixture for an in-tree CUDA regression. Every V lane
# is identical, allowing the pinned expected result to remain a small file.
t,hk,hv=129,2,6
r=torch.arange(t,device='cuda')[:,None,None]
kh=torch.arange(hk,device='cuda')[None,:,None]
d=torch.arange(128,device='cuda')[None,None,:]
q=(((r*7+kh*11+d*3)%31)-15).float()/16
k=(((r*5+kh*13+d*7)%29)-14).float()/16
vh=torch.arange(hv,device='cuda')[None,:,None]
v=((((r*3+vh*5)%23)-11).float()/8).expand(t,hv,128)
x=torch.cat((q.reshape(t,-1),k.reshape(t,-1),v.reshape(t,-1)),dim=1).to(torch.bfloat16)
rt=torch.arange(t,device='cuda')[:,None]
ht=torch.arange(hv,device='cuda')[None,:]
a=(-3+(((rt+2*ht)%13)-6).float()/8).to(torch.bfloat16)
b=((((rt*2+ht)%9)-4).float()/4).to(torch.bfloat16)
al=-3+torch.arange(hv,device='cuda').float()/8
dt=-2.5+torch.arange(hv,device='cuda').float()/16
st=((((torch.arange(hv,device='cuda')[:,None]*3+torch.arange(128,device='cuda')[None,:]*5)%17)-8).float()/32)
initial=st[None,:,None,:].expand(1,hv,128,128).contiguous()
eo,es=check('in-tree-regression',x,a,b,al,dt,initial,hk,hv)
old_o,old_s,_=native(x,a,b,al,dt,initial,hk,hv,recurrent=True)
Path('/output/regression-old-kernel.json').write_text(json.dumps({
    'output':stats(eo,old_o),'state':stats(es,old_s),
},indent=2)+'\n')
assert torch.equal(eo,eo[:,:,:,:1].expand_as(eo))
assert torch.equal(es,es[:,:,:1,:].expand_as(es))
Path('/output/regression-reference.json').write_text(json.dumps({
    'output_bf16':eo[0,:,:,0].contiguous().cpu().view(torch.uint16).tolist(),
    'state_f32':es[0,:,0,:].cpu().tolist(),
    'tokens':t,'key_heads':hk,'value_heads':hv,
    'source':'Pinned vLLM 8e685d198 FLA chunk_gated_delta_rule with fused_post_conv_prep',
},indent=2)+'\n')
from tuning import probe
probe(grouped,stats)
assert all(r['pass'] for r in results), 'Native chunk comparison exceeded predefined numerical budgets'
print('PASS',len(results),'native/reference comparisons',flush=True)
