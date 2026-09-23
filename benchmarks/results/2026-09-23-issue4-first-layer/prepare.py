#!/usr/bin/env python3
"""Construct first-layer GR and dense operands directly from tokens/checkpoint."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'2026-09-23-issue4-ple-projection/replay.py'
spec=importlib.util.spec_from_file_location('oracle',SOURCE)
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
bf,rb,norm,stats=oracle.bf,oracle.rb,oracle.norm,oracle.statistics
CHECKPOINT=Path('/home/stephen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47')
weights=oracle.Weights(CHECKPOINT)
config=json.loads((CHECKPOINT/'config.json').read_text())['text_config']
assert config['hc_count']==4 and config['hidden_size']==2560
tokens=np.asarray(json.loads((ROOT.parent/'2026-09-23-issue4-operators/raw/forced-prefix.ids.json').read_text()),dtype='<i8')
assert len(tokens)==261290 and hashlib.sha256(tokens.tobytes()).hexdigest()=='fe207c412e6a70d70b00bbc3c5995488ed12ce964c4ac7eb90933ab61e82785d'
unique,inverse=np.unique(tokens,return_inverse=True)
assert len(unique)==354
CAPTURE=ROOT.parent/'2026-09-23-issue4-operators/raw/captures'
output=ROOT/'raw/input'
output.mkdir(exist_ok=False)
inverse.astype('<i4').tofile(output/'token_rows.bin')
unique.astype('<i8').tofile(output/'unique_tokens.bin')
hashes={}
checks={}


def save(name,values,bfloat=True):
    if bfloat:
        values=(np.asarray(values,dtype='<f4').view('<u4')>>16).astype('<u2')
    else:
        values=np.asarray(values,dtype='<f4')
    values.tofile(output/(name+'.bin'))


def captured(rank,name,width,dtype='<u2'):
    path=CAPTURE/f'rank{rank}/layer0/{name}.bin'
    hashes[str(path.relative_to(ROOT.parent))]=oracle.sha(path)
    values=np.fromfile(path,dtype=dtype).reshape(-1,width)
    return bf(values) if dtype=='<u2' else values


def fp8(values):
    nr,nc=values.shape
    return oracle.block_fp8(np.pad(values,((0,(-nr)%128),(0,(-nc)%128))))[:nr,:nc]


def sigmoid(values):
    return 1/(1+np.exp(-np.asarray(values,dtype=np.float64)))


def embedding():
    name='model.language_model.embed_tokens.weight'
    with (CHECKPOINT/weights.index[name]).open('rb') as stream:
        size,=struct.unpack('<Q',stream.read(8))
        item=json.loads(stream.read(size))[name]
        assert item['dtype']=='BF16' and item['shape'][1]==2560
        values=[]
        for token in unique:
            stream.seek(8+size+item['data_offsets'][0]+int(token)*2560*2)
            payload=stream.read(2560*2)
            assert len(payload)==2560*2
            values.append(np.frombuffer(payload,dtype='<u2'))
    values=np.stack(values)
    weights.hashes[name+'[354 selected token rows]']=hashlib.sha256(values.tobytes()).hexdigest()
    return bf(values)


start=time.monotonic()
residual=np.tile(embedding(),(1,4))
prefix='model.language_model.layers.0.attn_hyper_connection.'
normalized=norm(residual,weights.read(prefix+'hc_norm.weight'),config['rms_norm_eps']).reshape(len(unique),10240)
down=fp8(weights.read(prefix+'input_mix_weight_down.weight'))
reduced=rb(normalized.astype(np.float64)@down.astype(np.float64).T)
scaled=reduced/np.float32(4)
activated=rb(scaled*sigmoid(scaled))
up=fp8(weights.read(prefix+'input_mix_weight_up.weight'))
logits=rb(activated.astype(np.float64)@up.astype(np.float64).T).reshape(len(unique),4,2560)
products=rb(rb(sigmoid(logits))*normalized.reshape(len(unique),4,2560))
attention=rb(products.sum(axis=1,dtype=np.float32)/np.float32(4))
save('attention_lookup',attention)
positions=list(range(260096,261290))
for rank in (0,1):
    checks[f'rank{rank}/attention_input']=stats(attention[inverse[-1194:]],captured(rank,'attention_input',2560),positions)
prefix='model.language_model.layers.0.linear_attn.'
qkv=weights.read(prefix+'in_proj_qkv.weight')
conv=weights.read(prefix+'conv1d.weight').reshape(10240,4)
z=weights.read(prefix+'in_proj_z.weight')
a=weights.read(prefix+'in_proj_a.weight')
b=weights.read(prefix+'in_proj_b.weight')
alog=weights.read(prefix+'A_log')
dt=weights.read(prefix+'dt_bias')
norm_weight=weights.read(prefix+'norm.weight')
out=weights.read(prefix+'out_proj.weight')
for rank in (0,1):
    selection=np.concatenate([np.arange(rank*1024,(rank+1)*1024),np.arange(2048+rank*1024,2048+(rank+1)*1024),np.arange(4096+rank*3072,4096+(rank+1)*3072)])
    matrices={'qkv':fp8(qkv[selection]),'z':fp8(z[rank*3072:(rank+1)*3072]),
              'a':a[rank*24:(rank+1)*24],'b':b[rank*24:(rank+1)*24]}
    for name,matrix in matrices.items():
        lookup=rb(attention.astype(np.float64)@matrix.astype(np.float64).T)
        save(f'{name}{rank}',lookup)
        checks[f'rank{rank}/{name}']=stats(lookup[inverse[-1194:]],captured(rank,name,matrix.shape[0]),positions)
    for name,values,dtype in [('conv_weight',conv[selection],'<u2'),('a_log',alog[rank*24:(rank+1)*24],'<f4'),
                               ('dt_bias',dt[rank*24:(rank+1)*24],'<f4'),('norm_weight',norm_weight,'<u2')]:
        actual=captured(rank,name,values.size,dtype).reshape(values.shape)
        assert np.array_equal(values.view('u4'),actual.view('u4')),(rank,name)
        save(f'{name}{rank}',values,bfloat=dtype=='<u2')
    save(f'output_weight{rank}',fp8(out[:,rank*3072:(rank+1)*3072]),bfloat=False)
    for name in ['state_before','state_after']:
        captured(rank,name,24*128*128,'<f4')
scale=np.float32(1)/np.sqrt(np.float32(128))
scalars=np.array([config['rms_norm_eps'],scale],dtype='<f4')
for rank in (0,1):
    assert np.array_equal(scalars,captured(rank,'gdn_scalars',2,'<f4').reshape(2))
scalars.tofile(output/'scalars.bin')
report=dict(completed=True,tokens=len(tokens),unique_tokens=len(unique),checks=checks,
            tensor_sha256=weights.hashes,captured_sha256=hashes,
            generated_sha256={p.name:oracle.sha(p) for p in output.iterdir()},
            seconds=time.monotonic()-start,script_sha256=oracle.sha(Path(__file__)),oracle_sha256=oracle.sha(SOURCE),
            scope='First-layer inputs generated from original token IDs and checkpoint; captures only comparisons/constants parity assertions. FP64 products, BF16 staging, native FP8 weight policy, original BF16 a/b weights.')
(ROOT/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
for name,check in checks.items():print(name,check['relative_l2'],flush=True)
