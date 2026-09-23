#!/usr/bin/env python3
"""Check the actual model adapter at TP2 GVA geometry before full model load."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule

root = Path('/inputs')
case = json.loads((root/'fixtures.json').read_text())[0]
path = root/'fixtures'/case['file']
assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
d = np.load(path)


def tensor(a, bf16=False):
    v = torch.from_numpy(a.copy())
    return (v.view(torch.bfloat16) if bf16 else v).cuda()


raw = tensor(d['qkv'],True)
# Replication tests the real grouped-value head mapping, not model accuracy.
x = torch.cat((raw[:,:128].repeat(1,8),raw[:,128:256].repeat(1,8),raw[:,256:].repeat(1,24)),dim=1)
a,b = (tensor(d[k],True).repeat(1,24) for k in ('a','b'))
al,dt = (tensor(d[k]).repeat(24) for k in ('alog','dt'))
prepared = fused_post_conv_prep(x,a,b,al,dt,8,128,128,apply_l2norm=True)
q,k,v,g,beta = (t.unsqueeze(0) for t in prepared)
initial = tensor(d['state_before']).repeat(1,24,1,1)
original = initial.clone()
cu = torch.tensor([0,x.shape[0]],device='cuda',dtype=torch.int32)
kwargs = dict(q=q,k=k,v=v,g=g,beta=beta,initial_state=initial,
              output_final_state=True,cu_seqlens=cu,use_qk_l2norm_in_kernel=False)
buffer = torch.empty_like(v)
out,state = ChunkGatedDeltaRule.forward_native(None,**kwargs,core_attn_out=buffer)
torch.cuda.synchronize()
assert torch.equal(initial,original), 'Adapter mutated input state'
assert torch.equal(out,buffer), 'Optional output buffer differs'
assert state.shape == initial.shape and out.shape == v.shape
assert bool(torch.isfinite(out).all() and torch.isfinite(state).all())
pool = torch.cat((torch.zeros_like(initial),initial),dim=0)
indices = torch.ones((1,x.shape[0]),device='cuda',dtype=torch.int32)
expected,expected_pool = fused_recurrent_gated_delta_rule(
    q=q,k=k,v=v,g=g,beta=beta,initial_state=pool,cu_seqlens=cu,
    inplace_final_state=True,ssm_state_indices=indices,use_qk_l2norm_in_kernel=False)
assert torch.equal(out,expected) and torch.equal(state,expected_pool[1:2])
torch.cuda.synchronize()
start = time.monotonic()
second,second_state = ChunkGatedDeltaRule.forward_native(None,**kwargs)
torch.cuda.synchronize()
seconds = time.monotonic()-start
assert torch.equal(out,second) and torch.equal(state,second_state)
assert torch.equal(initial,original)
result = {'pass':True,'tokens':x.shape[0],'key_heads':8,'value_heads':24,
          'input_state_unchanged':True,'optional_output_buffer_exact':True,
          'direct_operator_exact':True,'repeat_exact':True,
          'warm_wall_seconds':seconds,'scope':'Replicated frozen inputs test adapter shapes/state routing; not a model accuracy result.'}
Path('/output/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
