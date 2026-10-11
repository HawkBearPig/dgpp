import hashlib,json,pathlib,statistics,os
out=pathlib.Path(os.environ.get('PR95_OUT',str(pathlib.Path(__file__).parent)))
report={'prefill':[],'decode':[],'worlds':[]}
all_hashes=set()
for world in (1,2,4):
 legs=[]
 for suffix in ('fp8a','nvfp4','fp8b'):
  d=out/f'final-w{world}-{suffix}'
  for cmd in d.glob('*.command.json'):
   assert json.loads(cmd.read_text())['returncode']==0,cmd
  digest=(d/'binary.sha256').read_text().strip();all_hashes.add(digest)
  assert json.loads((d/'opstreams.json').read_text())['match'],d
  assert 'ALL OK []' in (d/'api.log').read_text(),d
  for when in ('before','after'):
   nodes=json.loads((d/f'nodes-{when}.json').read_text());assert len(nodes)==world
   assert all(n['binary_sha256']==digest and n['vm_swap_kib']==0 and n['cgroup_swap_current']=='0' and n['cgroup_swap_max']=='0' for n in nodes),d
  pre=json.loads((d/'prefill.json').read_text());dec=json.loads((d/'decode.json').read_text())
  assert pre['seed']==7 and pre['tag']=='pr95-matched'
  assert len(pre['samples'])==12 and len(dec['phases'])==60
  pm={}
  for s in pre['samples']:
   assert s['cached_tokens']==0 and s['computed_tokens']==s['prompt_tokens']
   assert s['usage']['completion_tokens']==1 and s['prefill_ms']>0
   key=(s['requested_tokens'],s['repeat']);assert key not in pm;pm[key]=s
  dm={}
  for phase in dec['phases']:
   key=(phase['class'],phase['concurrency']);dm.setdefault(key,[]).append(phase)
   req=phase['requests'];assert len(req)==phase['concurrency'] and all(r['status']==200 for r in req)
   assert all(0<r['usage']['completion_tokens']<=dec['max_tokens'] for r in req)
   assert phase['engine']['tokens_per_s']>0
  assert set(dm)=={(c,n) for c in ('prose','code','json','math','chat') for n in (1,2,4,8)}
  assert all(len(v)==3 for v in dm.values())
  legs.append((pm,dm))
 for length in (512,2048,8192,32768):
  samples=[[leg[0][length,r] for r in range(3)] for leg in legs]
  for r in range(3):
   assert len({(s[r]['prompt_sha256'],s[r]['prompt_tokens']) for s in samples})==1
  med=[statistics.median(s['prefill_ms'] for s in ss) for ss in samples]
  report['prefill'].append(dict(world=world,length=length,fp8a_ms=med[0],nvfp4_ms=med[1],fp8b_ms=med[2],reduction_pct=100*(1-med[1]/min(med[0],med[2])),fp8_drift_pct=100*(med[2]/med[0]-1),nvfp4_faster=med[1]<min(med[0],med[2]),paired_reduction_pct=[100*(1-samples[1][r]['prefill_ms']/min(samples[0][r]['prefill_ms'],samples[2][r]['prefill_ms'])) for r in range(3)]))
 for key in legs[0][1]:
  phases=[leg[1][key] for leg in legs]
  for r in range(3):
   assert len({tuple(x['prompt_sha256'] for x in ps[r]['requests']) for ps in phases})==1
  med=[statistics.median(p['engine']['tokens_per_s'] for p in ps) for ps in phases]
  report['decode'].append(dict(world=world,kind=key[0],concurrency=key[1],fp8a_tps=med[0],nvfp4_tps=med[1],fp8b_tps=med[2],gain_pct=100*(med[1]/max(med[0],med[2])-1),fp8_drift_pct=100*(med[2]/med[0]-1),nvfp4_faster=med[1]>max(med[0],med[2])))
 report['worlds'].append(world)
assert len(all_hashes)==1
report['binary_sha256']=all_hashes.pop();report['performance_gate']=all(r['nvfp4_faster'] for r in report['prefill']+report['decode'])
(out/'matched-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise SystemExit(0 if report['performance_gate'] else 1)
