import concurrent.futures,json,pathlib,shlex,subprocess
PROBE='''import hashlib,json,pathlib,subprocess,sys
state=json.loads(pathlib.Path(sys.argv[1]).read_text());pid=state['pid'];proc=pathlib.Path('/proc')/str(pid)
status=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
swap=int(status['VmSwap'].split()[0]);cg=(proc/'cgroup').read_text().strip();cgdir=pathlib.Path('/sys/fs/cgroup')/cg.split('::')[-1].lstrip('/')
read=lambda name:(cgdir/name).read_text().strip() if (cgdir/name).exists() else None
smi=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(pid=pid,binary_sha256=hashlib.sha256((proc/'exe').read_bytes()).hexdigest(),vm_swap_kib=swap,cgroup=cg,cgroup_swap_current=read('memory.swap.current'),cgroup_swap_max=read('memory.swap.max'),gpu=smi.stdout.strip())))
'''
def capture(directory,label):
 directory=pathlib.Path(directory);cfg=json.loads((directory/'world/cluster.resolved.json').read_text())
 def node(item):
  rank,host=item;state=str(directory/'world/rank0.process.json') if rank==0 else cfg['paths']['stage_dir']+f'/rank{rank}.process.json'
  cmd=['python3','-',state]
  if rank:cmd=['ssh','-o','BatchMode=yes',cfg['ssh_user']+'@'+host,shlex.join(cmd)]
  p=subprocess.run(cmd,input=PROBE,text=True,capture_output=True,check=True,timeout=30);return dict(rank=rank,host=host,**json.loads(p.stdout))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(node,enumerate(cfg['nodes'])))
 (directory/f'nodes-{label}.json').write_text(json.dumps(results,indent=2))
 expected=(directory/'binary.sha256').read_text().strip()
 assert all(x['binary_sha256']==expected and x['vm_swap_kib']==0 and x['cgroup_swap_current']=='0' and x['cgroup_swap_max']=='0' for x in results),results
 return results
