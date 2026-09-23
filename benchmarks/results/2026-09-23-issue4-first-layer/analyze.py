#!/usr/bin/env python3
"""Compare the independent first-layer history with retained native captures."""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('oracle',ROOT.parent/'2026-09-23-issue4-ple-projection/replay.py')
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
bf,rb,stats=oracle.bf,oracle.rb,oracle.statistics
INPUT=ROOT/'raw/input'
OUTPUT=ROOT/'raw/output'
CAPTURE=ROOT.parent/'2026-09-23-issue4-operators/raw/captures'
preparation=json.loads((ROOT/'preparation.json').read_text())
selfcheck=json.loads((ROOT/'selfcheck.json').read_text())
assert selfcheck['passed'] and oracle.sha(ROOT/'raw/recur')==selfcheck['executable_sha256']
for name,digest in preparation['generated_sha256'].items():assert oracle.sha(INPUT/name)==digest,name
tokens=np.fromfile(INPUT/'token_rows.bin',dtype='<i4')
assert len(tokens)==261290
positions=list(range(260096,261290))
eps=float(np.fromfile(INPUT/'scalars.bin',dtype='<f4')[0])
checks={}
head_checks=[]
hashes={}
partials=[]
start=time.monotonic()


def captured(rank,name,shape,dtype='<u2'):
    path=CAPTURE/f'rank{rank}/layer0/{name}.bin'
    checksum=oracle.sha(path)
    key=str(path.relative_to(ROOT.parent))
    if key in preparation['captured_sha256']:assert checksum==preparation['captured_sha256'][key],key
    hashes[key]=checksum
    value=np.fromfile(path,dtype=dtype).reshape(shape)
    return bf(value) if dtype=='<u2' else value


def generated(rank,name,shape,dtype='<u2'):
    path=OUTPUT/f'{name}{rank}.bin'
    hashes[str(path.relative_to(ROOT))]=oracle.sha(path)
    value=np.fromfile(path,dtype=dtype).reshape(shape)
    assert np.isfinite(value).all()
    return bf(value) if dtype=='<u2' else value


for rank in (0,1):
    for name in ['state_before','state_after']:
        expected=generated(rank,name,(24,128,128),'<f8')
        actual=captured(rank,name,(24,128,128),'<f4')
        # In state statistics, worst_rows.position identifies the global value head.
        checks[f'rank{rank}/{name}']=stats(expected,actual,list(range(rank*24,(rank+1)*24)))
    for name,where in [('conv_before',list(range(260093,260096))),('conv_after',list(range(261287,261290)))]:
        checks[f'rank{rank}/{name}']=stats(generated(rank,name,(5120,3)).T,captured(rank,name,(5120,3)).T,where)
    qkvc=generated(rank,'qkvc',(1194,5120))
    checks[f'rank{rank}/qkvc']=stats(qkvc,captured(rank,'qkvc',(1194,5120)),positions)
    core=generated(rank,'core',(1194,24,128))
    native_core=captured(rank,'core',core.shape)
    checks[f'rank{rank}/core']=stats(core,native_core,positions)
    for head in range(24):
        head_checks.append(dict(rank=rank,head=head,core=stats(core[:,head],native_core[:,head],positions)))
    z=bf(np.fromfile(INPUT/f'z{rank}.bin',dtype='<u2').reshape(354,24,128))[tokens[-1194:]]
    weight=bf(np.fromfile(INPUT/f'norm_weight{rank}.bin',dtype='<u2'))
    rstd=(1/np.sqrt(np.mean(core.astype(np.float64)**2,axis=-1,keepdims=True)+eps)).astype(np.float32)
    normalized=rb(rb(rb(core*rstd)*weight)*(1/(1+np.exp(-z.astype(np.float64)))))
    checks[f'rank{rank}/gated_norm']=stats(normalized,captured(rank,'normed',normalized.shape),positions)
    (normalized.view('u4')>>16).astype('<u2').tofile(OUTPUT/f'normed{rank}.bin')
    matrix=np.fromfile(INPUT/f'output_weight{rank}.bin',dtype='<f4').reshape(2560,3072)
    partial=rb(normalized.reshape(1194,3072).astype(np.float64)@matrix.astype(np.float64).T)
    partials.append(partial)
folded=rb(partials[0]+partials[1])
native_folded=captured(0,'attention_folded',folded.shape)
assert np.array_equal(native_folded,captured(1,'attention_folded',folded.shape))
checks['folded_attention_output']=stats(folded,native_folded,positions)
assert all(item['nonfinite']==0 for item in checks.values())
report=dict(completed=True,tokens_replayed_from_zero=261290,value_heads=48,layers=1,
            checkpoints=[260096,261290],final_chunk_rows_compared=1194,checks=checks,per_head_core=head_checks,
            captured_and_output_sha256=hashes,preparation_sha256=oracle.sha(ROOT/'preparation.json'),
            selfcheck_sha256=oracle.sha(ROOT/'selfcheck.json'),script_sha256=oracle.sha(Path(__file__)),
            analysis_seconds=time.monotonic()-start,
            scope='Original tokens and checkpoint -> GR input mix -> native-policy dense projections -> complete convolution/recurrent history from zero -> gated norm -> output projection/TP2 fold. No captured activations or state seed this computation. FP64 arithmetic retains specified BF16 storage boundaries; not bitwise GPU emulation or a complete model execution.')
(ROOT/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
for name,value in checks.items():print(name,'L2',value['relative_l2'],'max',value['max_abs'],flush=True)
