"""Temporary read-only trace of repeated issue #4 requests in pinned vLLM.

Copies values to CPU without changing model arithmetic. Synchronization can
affect a race: response parity must be checked before interpreting this trace.
"""
import hashlib
import json
from pathlib import Path

import torch

PREFIX = [248045, 8678, 198, 84642, 4581, 1328, 55616, 28203,
          494, 279, 16713, 45366, 13, 3301, 4566, 1132]
request = 0
position = -1
layer = -1
active = False
directory = None


def begin(input_ids, positions):
    global request, position, layer, active, directory
    active = False
    if input_ids is None or input_ids.numel() == 0:
        return
    tokens = input_ids.detach().reshape(-1).cpu()
    pos = positions.detach().reshape(-1, tokens.numel())[0].cpu()
    position = int(pos[0])
    if position == 0:
        if tokens.numel() < len(PREFIX) or tokens[:len(PREFIX)].tolist() != PREFIX:
            return  # Ignore synthetic profiling/warmup inputs.
        request += 1
    if request == 0 or position >= 261320:
        return
    active = True
    layer = -1
    rank = torch.distributed.get_rank()
    directory = Path('/capture') / f'rank{rank}' / f'request{request}' / f'position{position}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'input.json').write_text(json.dumps({
        'position': position, 'tokens': tokens.tolist(), 'positions': pos.tolist(),
    }) + '\n')


def set_layer(index):
    global layer
    layer = index


def capture(name, tensor):
    if not active or tensor is None:
        return
    original_shape = list(tensor.shape)
    # Full first chunk to expose early differences; final row of later chunks.
    sample = tensor if position in (0, 2048) and layer < 4 else tensor[-1:]
    host = sample.detach().contiguous().cpu()
    data = host.view(torch.uint8).numpy().tobytes()
    stem = f'layer{layer}_{name}'
    path = directory / (stem + '.bin')
    with path.open('xb') as output:
        output.write(data)
    (directory / (stem + '.json')).write_text(json.dumps({
        'dtype': str(tensor.dtype), 'original_shape': original_shape,
        'saved_shape': list(host.shape), 'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }) + '\n')



def config_receipt():
    if not active or layer != 0 or position not in (0, 2048, 260096):
        return
    import importlib
    from triton.runtime.autotuner import Autotuner
    names=[('chunk_scaled_dot_kkt','chunk_scaled_dot_kkt_fwd_kernel'),
           ('wy_fast','recompute_w_u_fwd_kernel'),
           ('chunk_delta_h','chunk_gated_delta_rule_fwd_kernel_h_blockdim64'),
           ('chunk_o','chunk_fwd_kernel_o'),
           ('solve_tril','solve_tril_16x16_kernel'),
           ('solve_tril','merge_16x16_to_32x32_inverse_kernel'),
           ('solve_tril','merge_16x16_to_64x64_inverse_kernel')]
    result={}
    for module,name in names:
        obj=getattr(importlib.import_module('vllm.third_party.flash_linear_attention.ops.'+module),name)
        while not isinstance(obj,Autotuner): obj=obj.fn
        result[name]={str(k):dict(kwargs=v.kwargs,num_warps=v.num_warps,num_stages=v.num_stages)
                      for k,v in obj.cache.items()}
    (directory/'gdn-kernel-configs.json').write_text(json.dumps(result,indent=2)+'\n')


def capture_gdn(q,k,v,g,beta,state):
    pin_kernel_configs_once()
    if not active:
        return
    for name,value in zip(('q','k','v','g','beta'),(q,k,v,g,beta)):
        capture('gdn_'+name,value.squeeze(0))
    if position in (0,2048) and layer < 4:
        capture('gdn_initial_state',state)


def capture_gdn_result(result):
    if not active:
        return
    capture('gdn_core',result[0].squeeze(0))
    if position in (0,2048) and layer < 4:
        capture('gdn_final_state',result[1])
    config_receipt()


_kernels_pinned = False

def pin_kernel_configs_once():
    global _kernels_pinned
    if _kernels_pinned:
        return
    import importlib
    from triton.runtime.autotuner import Autotuner
    modules={'chunk_scaled_dot_kkt_fwd_kernel':'chunk_scaled_dot_kkt',
             'recompute_w_u_fwd_kernel':'wy_fast',
             'chunk_gated_delta_rule_fwd_kernel_h_blockdim64':'chunk_delta_h',
             'chunk_fwd_kernel_o':'chunk_o',
             'solve_tril_16x16_kernel':'solve_tril',
             'merge_16x16_to_32x32_inverse_kernel':'solve_tril',
             'merge_16x16_to_64x64_inverse_kernel':'solve_tril'}
    rank=torch.distributed.get_rank()
    desired=json.loads(Path('/gdn-configs.json').read_text())[str(rank)]
    for name,entries in desired.items():
        if not entries:
            continue
        variants=list(entries.values())
        assert all(x==variants[0] for x in variants), (name,variants)
        want=variants[0]
        obj=getattr(importlib.import_module('vllm.third_party.flash_linear_attention.ops.'+modules[name]),name)
        while not isinstance(obj,Autotuner): obj=obj.fn
        found=[c for c in obj.configs if c.kwargs==want['kwargs'] and c.num_warps==want['num_warps'] and c.num_stages==want['num_stages']]
        assert len(found)==1,(name,want)
        obj.configs=found
        obj.cache.clear()
    _kernels_pinned=True
    print('Issue4: pinned GDN configurations to the measured BF16-state world, rank',rank,flush=True)


_moe_hooks_ready = False
_moe_capture_counts = {}
_marlin_bundle_count = 0

def moe_active():
    return active and layer == 0 and position == 0

def moe_capture(name,tensor):
    if moe_active():
        key=(request,name)
        index=_moe_capture_counts.get(key,0)+1
        _moe_capture_counts[key]=index
        capture('moe_'+name+('' if index==1 else f'_call{index}'),tensor)

def prepare_moe(module):
    global _moe_hooks_ready
    if not moe_active() or _moe_hooks_ready:
        return
    _moe_hooks_ready=True
    def gate_hook(mod,args,out):
        moe_capture('router_logits',out[0] if isinstance(out,tuple) else out)
    def shared_hook(mod,args,out):
        moe_capture('shared_output',out)
    module.gate.register_forward_hook(gate_hook)
    if module.shared_expert is not None:
        module.shared_expert.register_forward_hook(shared_hook)
    parameters={}
    for prefix,obj in [('gate',module.gate),('shared_gate',module.shared_expert_gate),('shared',module.shared_expert)]:
        if obj is not None:
            for name,p in obj.named_parameters():
                parameters[prefix+'.'+name]=p.detach().cpu().clone()
    torch.save(parameters,directory/'moe-dense-parameters.pt')

def save_marlin_bundle(args):
    global _marlin_bundle_count
    if not moe_active():
        return
    for name in ('hidden_states','topk_ids','topk_weights','sorted_token_ids','expert_ids','num_tokens_post_padded'):
        moe_capture(name,args.get(name))
    if request != 1:
        return
    tensor_names=('hidden_states','w1','w2','bias1','bias2','w1_scale','w2_scale',
        'topk_weights','topk_ids','global_scale1','global_scale2','input_global_scale1',
        'input_global_scale2','g_idx1','g_idx2','sort_indices1','sort_indices2',
        'w1_zeros','w2_zeros','expert_map','sorted_token_ids','expert_ids','num_tokens_post_padded')
    bundle={k:(args[k].detach().cpu().clone() if isinstance(args.get(k),torch.Tensor) else args.get(k)) for k in tensor_names}
    for k in ('quant_type_id','apply_router_weight_on_input','global_num_experts','activation','activation_config','input_dtype','is_k_full','block_size_m'):
        bundle[k]=args.get(k)
    bundle['activation_func_present']=args.get('activation_func') is not None
    bundle['moe_sum_present']=args.get('moe_sum') is not None
    _marlin_bundle_count+=1
    props=torch.cuda.get_device_properties(args['hidden_states'].device)
    bundle['device_info']={'name':props.name,'capability':[props.major,props.minor],'multiprocessors':props.multi_processor_count}
    name='marlin-bundle.pt' if _marlin_bundle_count==1 else f'marlin-bundle-{_marlin_bundle_count}.pt'
    torch.save(bundle,directory/name)
