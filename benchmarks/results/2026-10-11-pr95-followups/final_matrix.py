import datetime,hashlib,json,pathlib,subprocess,os
out=pathlib.Path(os.environ.get('PR95_OUT',str(pathlib.Path(__file__).parent)));root=pathlib.Path(__file__).resolve().parents[3]
out.mkdir(parents=True,exist_ok=True)
status=[]
for world in (1,2,4):
 for suffix,quant in [('fp8a','fp8'),('nvfp4','nvfp4'),('fp8b','fp8')]:
  label=f'final-w{world}-{suffix}';cmd=['python3',str(pathlib.Path(__file__).parent/'campaign.py'),quant,str(world),label,'--final']
  if world==4 and suffix in ('fp8a','nvfp4'):cmd+=['--quality','--quality-limit','100']
  e=dict(label=label,command=cmd,start=datetime.datetime.now(datetime.timezone.utc).isoformat());print(e['start'],label,flush=True)
  with (out/f'{label}-controller.log').open('w') as f:p=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
  e.update(returncode=p.returncode,end=datetime.datetime.now(datetime.timezone.utc).isoformat());status.append(e);(out/'final-matrix-status.json').write_text(json.dumps(status,indent=2))
  if p.returncode:raise SystemExit(p.returncode)
subprocess.run(['python3',str(pathlib.Path(__file__).parent/'audit_matched.py')],cwd=root,check=True)
