"""Bounded operator probe: change one FLA launch configuration at a time."""
import importlib
import json
from pathlib import Path

import torch
import numpy as np
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep


def kernels_for_probe():
    names=[('chunk_scaled_dot_kkt','chunk_scaled_dot_kkt_fwd_kernel'),
           ('wy_fast','recompute_w_u_fwd_kernel'),
           ('chunk_delta_h','chunk_gated_delta_rule_fwd_kernel_h_blockdim64'),
           ('chunk_o','chunk_fwd_kernel_o'),
           ('solve_tril','solve_tril_16x16_kernel'),
           ('solve_tril','merge_16x16_to_32x32_inverse_kernel'),
           ('solve_tril','merge_16x16_to_64x64_inverse_kernel')]
    kernels={}
    for module,name in names:
        obj=getattr(importlib.import_module('vllm.third_party.flash_linear_attention.ops.'+module),name)
        while not isinstance(obj,Autotuner): obj=obj.fn
        kernels[name]=obj
    return kernels


def desc(config):
    return dict(kwargs=config.kwargs,num_warps=config.num_warps,num_stages=config.num_stages)


def pin_baseline():
    kernels=kernels_for_probe()
    for name,obj in kernels.items():
        if not hasattr(obj,'_issue4_configs'): obj._issue4_configs=list(obj.configs)
        obj.configs=[obj._issue4_configs[0]]
        obj.cache.clear()
    Path('/output/reference-configs.json').write_text(json.dumps({name:desc(obj.configs[0]) for name,obj in kernels.items()},indent=2)+'\n')


def probe(inputs, stats):
    x,a,b,al,dt,initial,hk,hv=inputs
    q,k,v,g,beta=(z.unsqueeze(0) for z in fused_post_conv_prep(x,a,b,al,dt,hk,128,128,apply_l2norm=True))
    cu=torch.tensor([0,x.shape[0]],device='cuda',dtype=torch.int32)
    kernels=kernels_for_probe()
    original={name:list(getattr(obj,'_issue4_configs',obj.configs)) for name,obj in kernels.items()}
    cache={name:{str(k):desc(v) for k,v in obj.cache.items()} for name,obj in kernels.items()}
    for name,obj in kernels.items():
        obj.configs=[original[name][0]]
        obj.cache.clear()
    def run():
        out,st=chunk_gated_delta_rule(q,k,v,g,beta,initial_state=initial.clone(),output_final_state=True,
                                     cu_seqlens=cu,use_qk_l2norm_in_kernel=False)
        torch.cuda.synchronize()
        return out.clone(),st.clone()
    baseline=run()
    again=run()
    assert all(torch.equal(x,y) for x,y in zip(baseline,again))
    # Identify the precision used by the promoted beta*K operand in KKT.
    module=importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt')
    kk=k[:,:64,:1].contiguous()
    bb=beta[:,:64,:1].contiguous()
    actual=module.chunk_scaled_dot_kkt_fwd(k=kk,beta=bb,g=None)[0,:,0].cpu()
    right=kk[0,:,0].float().cpu().numpy()
    scaled=(kk[0,:,0].float()*bb[0,:,0,None]).cpu().numpy()
    bits=scaled.view(np.uint32)
    variants={'fp32':scaled,'tf32_truncate':(bits & np.uint32(0xffffe000)).view(np.float32),
              'tf32_nearest_even':((bits+np.uint32(4095)+((bits>>13)&1)) & np.uint32(0xffffe000)).view(np.float32),
              'tf32_nearest_away':((bits+np.uint32(4096)) & np.uint32(0xffffe000)).view(np.float32)}
    precision={name:stats(torch.from_numpy(np.tril((left.astype(np.float64)@right.astype(np.float64).T).astype(np.float32),-1)),actual)
               for name,left in variants.items()}
    changes=[]
    result=dict(tokens=x.shape[0],key_heads=hk,value_heads=hv,kernel_cache_before_probe=cache,
                kkt_precision_controls=precision,
                complete=False,fixed_config_repeat_exact=True,changes=changes,
                scope='Identical frozen inputs; does not establish which configurations ran in earlier full-model worlds.')
    Path('/output/autotune-variation.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,obj in kernels.items():
        unsupported=[]
        for config in reversed(original[name]):
            obj.configs=[config]
            obj.cache.clear()
            try:
                changed=run()
                break
            except OutOfResources as error:
                unsupported.append(dict(config=desc(config),error=str(error)))
        else:
            raise RuntimeError('No supported configuration for '+name)
        changes.append(dict(kernel=name,baseline=desc(original[name][0]),changed=desc(config),
                            unsupported=unsupported,output=stats(baseline[0],changed[0]),state=stats(baseline[1],changed[1])))
        obj.configs=[original[name][0]]
        obj.cache.clear()
        Path('/output/autotune-variation.json').write_text(json.dumps(result,indent=2)+'\n')
    result['complete']=True
    Path('/output/autotune-variation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed FLA launch-configuration probe',flush=True)
