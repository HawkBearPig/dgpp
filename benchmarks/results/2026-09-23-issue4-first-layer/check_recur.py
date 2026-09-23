#!/usr/bin/env python3
"""Cross-check the standalone C++ recurrence against vectorized NumPy."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('oracle',ROOT.parent/'2026-09-23-issue4-ple-projection/replay.py')
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
bf,rb=oracle.bf,oracle.rb
source=ROOT/'raw/input'
test=ROOT/'raw/selfcheck-input'
out=ROOT/'raw/selfcheck-output'
test.mkdir(exist_ok=False)
manifest=json.loads((ROOT/'preparation.json').read_text())['generated_sha256']
for path in source.iterdir():
    assert oracle.sha(path)==manifest[path.name]
    if path.name!='token_rows.bin':os.link(path,test/path.name)
tokens=np.fromfile(source/'token_rows.bin',dtype='<i4')[:1201]
assert len(tokens)==1201
tokens.tofile(test/'token_rows.bin')
run=subprocess.run([str(ROOT/'raw/recur'),str(test),str(out)],env=dict(os.environ,OMP_NUM_THREADS='2'),capture_output=True,text=True,check=True,timeout=180)
(ROOT/'raw/selfcheck.log').write_text(run.stdout+run.stderr)
scale=float(np.fromfile(source/'scalars.bin',dtype='<f4')[1])
reports=[]
for rank in (0,1):
    def values(name,width):return bf(np.fromfile(source/f'{name}{rank}.bin',dtype='<u2')).astype(np.float64).reshape(-1,width)
    lookup=values('qkv',5120)
    a,b=values('a',24),values('b',24)
    weight=values('conv_weight',4)
    alog=np.fromfile(source/f'a_log{rank}.bin',dtype='<f4').astype(np.float64)
    dt=np.fromfile(source/f'dt_bias{rank}.bin',dtype='<f4').astype(np.float64)
    state=np.zeros((24,128,128),dtype=np.float64)
    history=np.zeros((3,5120),dtype=np.float64)
    core=[];convolved=[]
    for t,index in enumerate(tokens):
        if t==7:before=state.copy();conv_before=history.T.copy()
        raw=lookup[index]
        mixed=raw*weight[:,3]+(history*weight[:,:3].T).sum(axis=0)
        conv=rb(mixed/(1+np.exp(-mixed))).astype(np.float64)
        history[:-1]=history[1:];history[-1]=raw
        query=np.repeat(conv[:1024].reshape(8,128),3,axis=0)
        key=np.repeat(conv[1024:2048].reshape(8,128),3,axis=0)
        value=conv[2048:].reshape(24,128)
        query=query*(scale/np.sqrt((query*query).sum(axis=1,keepdims=True)+1e-6))
        key=key/np.sqrt((key*key).sum(axis=1,keepdims=True)+1e-6)
        av=a[index]+dt
        decay=np.exp(-np.exp(alog)*np.where(av>20,av,np.log1p(np.exp(av))))
        beta=1/(1+np.exp(-b[index]))
        state*=decay[:,None,None]
        delta=(value-np.einsum('hvk,hk->hv',state,key))*beta[:,None]
        state+=delta[:,:,None]*key[:,None,:]
        result=rb(np.einsum('hvk,hk->hv',state,query))
        if t>=7:core.append(result);convolved.append(conv)
    maximum=0
    for name,expected in [('state_before',before),('state_after',state)]:
        got=np.fromfile(out/f'{name}{rank}.bin',dtype='<f8').reshape(expected.shape)
        error=float(np.max(np.abs(got-expected)))
        assert np.allclose(got,expected,rtol=1e-12,atol=1e-12),(rank,name,error)
        maximum=max(maximum,error)
    for name,expected in [('conv_before',conv_before),('conv_after',history.T),('qkvc',np.array(convolved)),('core',np.array(core))]:
        got=bf(np.fromfile(out/f'{name}{rank}.bin',dtype='<u2')).reshape(expected.shape)
        assert np.array_equal(got,expected),(rank,name,np.count_nonzero(got!=expected))
    reports.append(dict(rank=rank,heads=24,tokens=1201,checkpoint_position=7,state_max_abs_error=maximum,all_bf16_outputs_exact=True))
report=dict(passed=True,cases=reports,executable_sha256=oracle.sha(ROOT/'raw/recur'),
            cpp_sha256=oracle.sha(ROOT/'recur.cpp'),script_sha256=oracle.sha(Path(__file__)))
(ROOT/'selfcheck.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
