#!/usr/bin/env python3
"""Replay exported, prepared Marlin operands without loading the complete model."""
import argparse,hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe,_fused_marlin_moe
from vllm.scalar_type import ScalarType


def capture_tensor(root,name):
    p=root/('layer0_moe_'+name+'.json');m=json.loads(p.read_text());raw=p.with_suffix('.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==m['sha256'] and len(raw)==m['bytes']
    if m['dtype']=='torch.bfloat16':
        x=torch.from_numpy(np.frombuffer(raw,dtype='<u2').copy()).view(torch.bfloat16)
    else:
        dtype={'torch.float32':'<f4','torch.int32':'<i4','torch.int64':'<i8'}[m['dtype']]
        x=torch.from_numpy(np.frombuffer(raw,dtype=dtype).copy())
    return x.reshape(m['saved_shape'])


def digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def stats(actual,expected):
    x,y=actual.float().numpy().astype(np.float64),expected.float().numpy().astype(np.float64)
    assert x.shape==y.shape,(x.shape,y.shape)
    return dict(bitwise_exact=torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8)),different_elements=int(np.count_nonzero(x!=y)),
        elements=int(x.size),finite=bool(np.isfinite(x).all()),
        relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)),max_absolute=float(np.max(np.abs(x-y))))


def sum_experts(value,out,ids,expert_map):
    if expert_map is None: ops.moe_sum(value,out)
    else: ops.moe_sum(value,out,ids,expert_map)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--repetitions',type=int,default=10);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();torch.set_grad_enabled(False)
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as marlin_module
    from issue4_moe_align import moe_align_block_size
    marlin_module.moe_align_block_size=moe_align_block_size
    bundle=torch.load(args.capture/'marlin-bundle.pt',map_location='cpu',weights_only=False)
    props=torch.cuda.get_device_properties(0);actual_device=dict(name=props.name,capability=[props.major,props.minor],multiprocessors=props.multi_processor_count)
    assert actual_device==bundle['device_info'],(actual_device,bundle['device_info'])
    assert bundle['moe_sum_present'] and bundle['activation_func_present']
    gpu={k:(v.to('cuda') if isinstance(v,torch.Tensor) else v) for k,v in bundle.items()}
    public_keys=inspect.signature(fused_marlin_moe).parameters
    public={k:v for k,v in gpu.items() if k in public_keys}
    public['moe_sum']=sum_experts
    # The stock activation helper, using the captured activation and configuration,
    # is checked against the captured intermediate in the fixed-assignment replay.
    public['activation_func']=None
    fixed_keys=inspect.signature(_fused_marlin_moe).parameters
    fixed={k:v for k,v in gpu.items() if k in fixed_keys}
    fixed['quant_type']=ScalarType.from_id(bundle['quant_type_id'])
    fixed['num_topk']=bundle['topk_ids'].shape[1]
    expected_sum=capture_tensor(args.capture,'routed_sum')
    expected_down=capture_tensor(args.capture,'expert_down')
    expected_gate_up=capture_tensor(args.capture,'gate_up')
    expected_activation=capture_tensor(args.capture,'activation')
    report={'device':actual_device,'input_sha256':digest(bundle['hidden_states']),
            'captured_shapes':{k:list(v.shape) for k,v in bundle.items() if isinstance(v,torch.Tensor)},'runs':[]}
    for mode in ('canonical_assignment','frozen_assignment'):
        previous=None;hashes=[]
        for repeat in range(args.repetitions):
            out=torch.empty_like(gpu['hidden_states'])
            if mode=='canonical_assignment':
                fused_marlin_moe(**public,output=out)
                down=None
            else:
                gate_up_seen=[];act_seen=[]
                from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation, ApplyMoEActivationConfig
                def activation_probe(activation,output,input,topk_ids,expert_map):
                    gate_up_seen.append(input.detach().cpu().clone())
                    apply_moe_activation(activation,output,input,activation_config=(gpu['activation_config'] if gpu['activation_config'] is not None else ApplyMoEActivationConfig()),topk_ids=topk_ids,expert_map=expert_map)
                    act_seen.append(output.detach().cpu().clone())
                down=_fused_marlin_moe(**fixed,activation_func=activation_probe)
                sum_experts(down.view(gpu['hidden_states'].shape[0],fixed['num_topk'],-1),out,gpu['topk_ids'],gpu['expert_map'])
            torch.cuda.synchronize();host=out.cpu();sha=digest(host);hashes.append(sha)
            row=dict(mode=mode,repetition=repeat,output_sha256=sha,versus_captured=stats(host,expected_sum))
            if previous is not None: row['versus_first_replay']=stats(host,previous)
            else:
                previous=host.clone();torch.save(previous,args.output/(mode+'-first.pt'))
            if down is not None:
                row['expert_down_vs_captured']=stats(down.cpu(),expected_down)
                row['gate_up_vs_captured']=stats(gate_up_seen[0],expected_gate_up)
                row['activation_vs_captured']=stats(act_seen[0],expected_activation)
            report['runs'].append(row)
            (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        report[mode+'_unique_outputs']=len(set(hashes))
    report['wall_seconds']=time.monotonic()-start;report['complete']=True
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','captured_shapes')},indent=2))

if __name__=='__main__':main()
