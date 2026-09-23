#!/usr/bin/env python3
"""Independent final-row GR mix/injection boundary replay from saved operands."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
source = RESULTS / '2026-09-23-issue4-ple-projection/replay.py'
spec = importlib.util.spec_from_file_location('ple_oracle',source)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
bf, rb, norm, statistics = oracle.bf, oracle.rb, oracle.norm, oracle.statistics
checkpoint = Path('/home/stephen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47')
weights = oracle.Weights(checkpoint)
config = json.loads((checkpoint/'config.json').read_text())['text_config']
assert config['hc_count']==4 and config['hidden_size']==2560 and config['hc_lowrank']==320
assert config['ple_layer_ids']==[2]
captures = RESULTS / '2026-09-23-issue4-operators/raw/captures'
moe = RESULTS / '2026-09-23-issue4-moe-audit/raw/captures'
tokens = json.loads((RESULTS/'2026-09-23-issue4-operators/raw/forced-prefix.ids.json').read_text())
assert len(tokens)==261290
assert hashlib.sha256(np.asarray(tokens,dtype='<i8').tobytes()).hexdigest()=='fe207c412e6a70d70b00bbc3c5995488ed12ce964c4ac7eb90933ab61e82785d'
input_hashes = {}


def tail(path,width):
    with path.open('rb') as stream:
        stream.seek(-width*2,2)
        payload=stream.read(width*2)
    assert len(payload)==width*2
    input_hashes[str(path.relative_to(RESULTS))]=dict(file_bytes=path.stat().st_size,tail_sha256=hashlib.sha256(payload).hexdigest())
    return bf(np.frombuffer(payload,dtype='<u2'))


def same_ranks(paths,width):
    values=[tail(path,width) for path in paths]
    assert np.array_equal(values[0].view('u4'),values[1].view('u4')),paths
    return values[0]


def op(layer,field,width):
    return same_ranks([captures/f'rank{rank}/layer{layer}/{field}.bin' for rank in (0,1)],width)


def moe_row(layer,field):
    return same_ranks([moe/f'rank{rank}/position260096/layer{layer}/row1193_{field}.bin' for rank in (0,1)],2560)


def embedding(token):
    name='model.language_model.embed_tokens.weight'
    path=checkpoint/weights.index[name]
    with path.open('rb') as stream:
        size,=struct.unpack('<Q',stream.read(8))
        meta=json.loads(stream.read(size))[name]
        assert meta['dtype']=='BF16' and meta['shape'][1]==2560
        stream.seek(8+size+meta['data_offsets'][0]+token*2560*2)
        payload=stream.read(2560*2)
    weights.hashes[name+f'[row={token}]']=hashlib.sha256(payload).hexdigest()
    return np.tile(bf(np.frombuffer(payload,dtype='<u2')),4)


def fp8(values):
    # Low rank 320 has a partial final block. Match its zero-padded amax.
    rows,cols=values.shape
    padded=np.pad(values,((0,(-rows)%128),(0,(-cols)%128)))
    return oracle.block_fp8(padded)[:rows,:cols]


def sigmoid(values):
    return 1/(1+np.exp(-np.asarray(values,dtype=np.float64)))


def mix(residual,prefix):
    normalized=norm(residual,weights.read(prefix+'hc_norm.weight'),config['rms_norm_eps']).reshape(10240)
    down=fp8(weights.read(prefix+'input_mix_weight_down.weight'))
    reduced=rb(down.astype(np.float64)@normalized.astype(np.float64))
    scaled=reduced/np.float32(4)
    activated=rb(scaled*sigmoid(scaled))
    up=fp8(weights.read(prefix+'input_mix_weight_up.weight'))
    logits=rb(up.astype(np.float64)@activated.astype(np.float64)).reshape(4,2560)
    products=rb(rb(sigmoid(logits))*normalized.reshape(4,2560))
    return rb(products.sum(axis=0,dtype=np.float32)/np.float32(4)),normalized


def inject(residual,normalized,branch,prefix):
    matrix=weights.read(prefix+'block_inject_weight.weight')
    assert matrix.shape==(4,10240)
    dots=rb(matrix.astype(np.float64)@normalized.astype(np.float64))
    gates=np.float32(2)*rb(sigmoid(dots/np.float32(4)))
    update=rb(gates[:,None]*branch[None,:])
    return rb(residual.reshape(4,2560)+update).reshape(10240),gates


start=time.monotonic()
results=[]
for layer in range(48):
    if layer==0:
        before=embedding(tokens[-1])
    elif layer==1:
        before=same_ranks([ROOT/f'raw/ple-after-rank{rank}.bin' for rank in (0,1)],10240)
    else:
        before=op(layer-1,'post_mlp_residual',10240)
    prefix=f'model.language_model.layers.{layer}.'
    attention_input,attention_normalized=mix(before,prefix+'attn_hyper_connection.')
    after_attention,attention_gates=inject(before,attention_normalized,op(layer,'attention_folded',2560),prefix+'attn_hyper_connection.')
    mlp_input,mlp_normalized=mix(after_attention,prefix+'mlp_hyper_connection.')
    after_mlp,mlp_gates=inject(after_attention,mlp_normalized,moe_row(layer,'moe_folded'),prefix+'mlp_hyper_connection.')
    comparisons={
        'attention_mix':statistics(attention_input,op(layer,'attention_input',2560),[261289]),
        'attention_inject_to_mlp_mix':statistics(mlp_input,moe_row(layer,'input'),[261289]),
        'both_injections_to_post_mlp':statistics(after_mlp,op(layer,'post_mlp_residual',10240),[261289]),
    }
    assert all(value['nonfinite']==0 for value in comparisons.values())
    results.append(dict(layer=layer,comparisons=comparisons,attention_gates=attention_gates.tolist(),mlp_gates=mlp_gates.tolist()))
    print(layer,{key:round(value['relative_l2'],8) for key,value in comparisons.items()},flush=True)
summary={key:dict(elements=sum(row['comparisons'][key]['elements'] for row in results),
                  worst_layer=max(results,key=lambda row:row['comparisons'][key]['relative_l2'])['layer'],
                  worst_relative_l2=max(row['comparisons'][key]['relative_l2'] for row in results),
                  bitwise_equal=sum(row['comparisons'][key]['bitwise_equal_elements'] for row in results)) for key in results[0]['comparisons']}
report=dict(completed=True,position=261289,layers=48,rank_fields_bitwise_equal=True,
            results=results,summary=summary,elapsed_seconds=time.monotonic()-start,
            tensor_sha256=weights.hashes,input_tail_sha256=input_hashes,
            script_sha256=oracle.sha(Path(__file__)),oracle_sha256=oracle.sha(source),
            scope='Independent checkpoint GR chains using captured prior residual, folded attention and folded MoE. Final row only; combines separately parity-validated deterministic captures. Not a full independent model execution or a retrieval fix.')
(ROOT/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
