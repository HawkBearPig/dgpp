"""PR 95 review measurements; run sequentially on the reserved cluster."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, sys
from node_snapshot import capture
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=pathlib.Path(os.environ.get('PR95_OUT',str(pathlib.Path(__file__).parent)))
OUT.mkdir(parents=True,exist_ok=True)
ENV={**os.environ,'CUDA_DEVICE_MAX_CONNECTIONS':'32','DGPP_NO_SWAP':'1'}
# Loopback tests use two local NICs; serving uses the persisted per-node site settings.
for key in ('DGPP_ROCE_DEVICES','DGPP_ROCE_GID_INDICES'):
    ENV.pop(key,None)
BIN=pathlib.Path(os.environ.get('PR95_BIN',str(ROOT/'build-pr95/dgpp-serve')))
def run(args,path,timeout=1800,check=True):
    receipt={'command':list(map(str,args)),'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    print(receipt['started'],path,flush=True)
    with path.open('w') as f:
        try:
            result=subprocess.run(receipt['command'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
            receipt['returncode']=result.returncode
        except subprocess.TimeoutExpired:
            receipt['returncode']=124
    receipt['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.with_suffix(path.suffix+'.command.json').write_text(json.dumps(receipt,indent=2))
    if check and receipt['returncode']:raise RuntimeError(str(path)+' failed: '+str(receipt['returncode']))
    return receipt['returncode']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('quant',choices=['fp8','nvfp4']);ap.add_argument('world',type=int,choices=[1,2,4]);ap.add_argument('label');ap.add_argument('--lite',action='store_true');ap.add_argument('--final',action='store_true');ap.add_argument('--prefill-only',action='store_true');ap.add_argument('--profile',action='store_true');ap.add_argument('--prefill-profile',action='store_true');ap.add_argument('--tune',action='store_true');ap.add_argument('--plain',action='store_true');ap.add_argument('--idle-budget',type=int);ap.add_argument('--base-ms',type=float);ap.add_argument('--row-ms',type=float);ap.add_argument('--block-budget',type=int);ap.add_argument('--schedule-off',action='store_true');ap.add_argument('--fixed-depth',type=int);ap.add_argument('--draft-weights',choices=['fp8','checkpoint']);ap.add_argument('--prefetch-window',type=int);ap.add_argument('--prefetch-rate');ap.add_argument('--prefetch-form');ap.add_argument('--draft-probe',action='store_true');ap.add_argument('--probe-classes',default='prose');ap.add_argument('--probe-concurrency',default='1');ap.add_argument('--probe-repeat',type=int,default=3);ap.add_argument('--quality',action='store_true');ap.add_argument('--quality-limit',type=int,default=100)
    a=ap.parse_args();d=OUT/a.label;d.mkdir(exist_ok=False)
    c=json.loads((ROOT/f'deploy/cluster_qwen3.8-27b_{a.quant}_w{a.world}.example.json').read_text())
    c['http']={'bind_host':'127.0.0.1','port':18081};c['engine']['lookup_draft']=False
    if a.plain:c['engine'].update(dflash_model='',mtp=False,mtp_schedule=False)
    if a.idle_budget:c['engine']['prefill_idle_budget_tokens']=a.idle_budget
    if a.base_ms is not None:c['engine']['mtp_schedule_base_ms']=a.base_ms
    if a.row_ms is not None:c['engine']['mtp_schedule_row_ms']=a.row_ms
    if a.schedule_off or a.block_budget is not None:c['engine']['mtp_schedule']=False
    if a.block_budget is not None:c['engine']['dflash_batch_rows']=a.block_budget
    if a.fixed_depth is not None:c['engine'].update(mtp_schedule=True,mtp_schedule_min_depth=a.fixed_depth,mtp_schedule_lambda=1.0)
    if a.draft_weights:c['engine']['dflash_weights']=a.draft_weights
    if a.prefetch_window is not None:c['engine']['l2_prefetch_boundary_window_mib']=a.prefetch_window
    if a.prefetch_rate:c['engine']['l2_prefetch_boundary_rate']=a.prefetch_rate
    if a.prefetch_form:c['engine']['l2_prefetch_form']=a.prefetch_form
    config=d/'deployment.json';config.write_text(json.dumps(c,indent=2))
    (d/'binary.sha256').write_text(hashlib.sha256(BIN.read_bytes()).hexdigest()+'\n')
    up=['python3','scripts/dgpp-cluster','up','--config',config,'--bin',BIN,'--log-dir',d/'world']
    if a.profile or a.prefill_profile:up+=['--head-wrap',f'nsys profile -t cuda --cuda-graph-trace=node -o {d}/r0 --force-overwrite true']
    try:
        run(up,d/'up.log')
        if not (a.profile or a.prefill_profile):
            pid=int((d/'world/r0.pid').read_text().strip())
            live=hashlib.sha256(pathlib.Path(f'/proc/{pid}/exe').read_bytes()).hexdigest()
            (d/'live-binary.sha256').write_text(live+'\n')
            assert live==(d/'binary.sha256').read_text().strip(), 'live server binary mismatch'
            capture(d,'before')
        run(['python3','scripts/serve_bench.py','127.0.0.1','18081','32','warm'],d/'warm.log')
        if a.profile:
            run(['python3','scripts/timed_load.py','127.0.0.1','18081','--concurrency','1','--classes','code','--max-tokens','400','--repeat','1','--json-out',d/'decode.json'],d/'decode.log')
        elif a.prefill_profile:
            run(['python3','scripts/serve_prefill_probe.py','127.0.0.1','18081','8192','--repeat','1','--no-think','--tag','pr95-profile','--interval','1','--json-out',d/'prefill.json'],d/'prefill.log')
        elif a.draft_probe:
            run(['python3','scripts/timed_load.py','127.0.0.1','18081','--concurrency',a.probe_concurrency,'--classes',a.probe_classes,'--max-tokens','256','--repeat',str(a.probe_repeat),'--json-out',d/'decode.json'],d/'decode.log')
        elif a.prefill_only:
            run(['python3','scripts/serve_prefill_probe.py','127.0.0.1','18081','512','2048','8192','32768','--repeat','3','--no-think','--tag','pr95-matched','--json-out',d/'prefill.json'],d/'prefill.log')
        else:
            run(['python3','scripts/timed_load.py','127.0.0.1','18081','--concurrency','1,2,4,8' if a.final else ('1' if a.lite else '1,4'),'--classes','all' if a.final else 'prose,code','--max-tokens','256' if a.final else '128','--repeat','1' if a.lite else '3','--json-out',d/'decode.json'],d/'decode.log')
            if not a.tune:
                run(['python3','scripts/serve_prefill_probe.py','127.0.0.1','18081','512','2048','8192','32768','--repeat','1' if a.lite else '3','--no-think','--tag','pr95-matched','--json-out',d/'prefill.json'],d/'prefill.log')
            if not a.lite:
                run(['python3','scripts/serve_greedy_transcript.py','127.0.0.1','18081','--max-tokens','128','--out',d/'transcripts.json'],d/'transcripts.log')
                if not a.tune:
                    run(['python3','scripts/serve_load.py','127.0.0.1','18081','--concurrency','1,4','--classes','code','--max-tokens','128','--temperature','0.7','--json-out',d/'sampled.json'],d/'sampled.log')
                    run(['python3','scripts/serve_api_check.py','127.0.0.1','18081'],d/'api.log')
            if a.quality:
                run(['python3','scripts/serve_eval.py','127.0.0.1','18081','--out',d/'quality','--tasks','gsm8k,extract','--limit',str(a.quality_limit),'--concurrency','4','--max-tokens','4096','--no-think','--seed','20261010'],d/'quality.log')
        if not (a.profile or a.prefill_profile):capture(d,'after')
    finally:
        run(['python3','scripts/dgpp-cluster','down','--config',config,'--log-dir',d/'world'],d/'down.log',check=False)
        ops=sorted((d/'world').glob('serve_rank*.ops'))
        hashes={p.name:hashlib.md5(p.read_bytes()).hexdigest() for p in ops}
        (d/'opstreams.json').write_text(json.dumps({'hashes':hashes,'expected_ranks':a.world,'match':len(hashes)==a.world and len(set(hashes.values()))==1},indent=2))
    if len(hashes)!=a.world or len(set(hashes.values()))!=1:
        raise RuntimeError('rank operation streams did not match')
    if a.profile:
        run(['python3','scripts/nsys_step_breakdown.py',d/'r0.nsys-rep','--marker','publish_seq_kernel','--top','30'],d/'breakdown.txt')
    if a.prefill_profile:
        run(['python3','scripts/nsys_step_breakdown.py',d/'r0.nsys-rep','--burst','--top','30'],d/'breakdown.txt')
if __name__=='__main__':main()
