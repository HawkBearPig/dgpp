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
