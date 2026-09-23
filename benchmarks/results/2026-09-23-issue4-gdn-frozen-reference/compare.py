#!/usr/bin/env python3
"""Pinned reference recurrent/chunked operators on identical real GDN inputs."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule

ROOT = Path('/inputs')
results = []


def tensor(array, bf16=False):
    t = torch.from_numpy(array.copy())
    if bf16:
        t = t.view(torch.bfloat16)
    return t.cuda()


def stats(reference, actual):
    r, a = reference.double(), actual.double()
    return {'relative_l2': (torch.linalg.vector_norm(r-a)/torch.linalg.vector_norm(r).clamp_min(1e-30)).item(),
            'max_abs': (r-a).abs().max().item(),
            'different': (r != a).sum().item(), 'elements': r.numel(),
            'finite': bool(torch.isfinite(r).all() and torch.isfinite(a).all())}


for case in json.loads((ROOT/'fixtures.json').read_text()):
    path = ROOT/'fixtures'/case['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
    data = np.load(path)
    x, a, b = (tensor(data[name], True) for name in ('qkv','a','b'))
    al, dt = tensor(data['alog']), tensor(data['dt'])
    initial = tensor(data['state_before'])
    expected = tensor(data['core'], True).reshape(1,-1,1,128)
    expected_state = tensor(data['state_after'])
    cu = torch.tensor([0, x.shape[0]], device='cuda', dtype=torch.int32)
    scale = float(data['scale'][0])
    outputs = {}
    timings = {}
    for mode in ('recurrent_fp32_normalization','recurrent_bf16_normalization','chunk_bf16_normalization'):
        normalized = mode != 'recurrent_fp32_normalization'
        prepared = fused_post_conv_prep(x,a,b,al,dt,1,128,128,apply_l2norm=normalized)
        q,k,v,g,beta = (t.unsqueeze(0) for t in prepared)
        state = initial.clone()
        torch.cuda.synchronize()
        start = time.monotonic()
        kwargs = dict(q=q,k=k,v=v,g=g,beta=beta,scale=scale,initial_state=state,
                      cu_seqlens=cu,use_qk_l2norm_in_kernel=not normalized)
        if mode.startswith('chunk'):
            out, final = chunk_gated_delta_rule(**kwargs,output_final_state=True)
        else:
            # The serving recurrent kernel reserves state index zero and
            # requires an explicit valid slot for each in-place token update.
            kwargs['initial_state'] = torch.cat((torch.zeros_like(state),state),dim=0)
            indices = torch.ones((1,x.shape[0]),device='cuda',dtype=torch.int32)
            out, pool = fused_recurrent_gated_delta_rule(
                **kwargs,inplace_final_state=True,ssm_state_indices=indices)
            final = pool[1:2]
        torch.cuda.synchronize()
        timings[mode] = time.monotonic()-start
        outputs[mode] = out.clone(), final.clone()
    raw, rounded, chunk = (outputs[k] for k in outputs)
    checks = {'reference_recurrent_vs_captured_core':stats(expected,raw[0]),
              'reference_recurrent_vs_captured_state':stats(expected_state,raw[1]),
              'normalized_storage_output':stats(raw[0],rounded[0]),
              'normalized_storage_state':stats(raw[1],rounded[1]),
              'chunk_vs_recurrent_output':stats(rounded[0],chunk[0]),
              'chunk_vs_recurrent_state':stats(rounded[1],chunk[1])}
    assert all(v['finite'] for v in checks.values())
    record = dict(case=case,checks=checks,timings_seconds_including_first_compile=timings)
    results.append(record)
    print(json.dumps(record),flush=True)
    Path('/output/comparison.json').write_text(json.dumps(results,indent=2)+'\n')
print('Completed',len(results),'frozen cases',flush=True)
