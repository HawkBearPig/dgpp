#!/usr/bin/env python3
"""Serial exact-fixture capture with production restoration in finally."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import time
import shlex
import importlib.util

ROOT = Path('/home/stephen/workspace/dgpp')
WORK = Path('/tmp/dgpp-issue4-20260923')
RECORD = Path(__file__).resolve().parent
RAW = RECORD / 'raw'
PROD = ROOT / 'deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json'
CAPTURE = Path('/tmp/dgpp-issue4-history-capture')
TEST_ENV = dict(os.environ, DGPP_ENV_FILE=str(WORK / 'benchmarks/results/2026-09-23-issue4-exact/raw/site.env'))
PROD_ENV = dict(os.environ, DGPP_ENV_FILE=str(ROOT / '.env'))
EXPECTED_PROD = '4638e3c595a2765d6d50c4626a9fef3aa2aa82936c6c50e3d0eec871fcde8e00'
receipts = []


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(name, command, env=PROD_ENV, cwd=ROOT, timeout=1200):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), name, 'start', flush=True)
    with (RAW / (name + '.log')).open('w') as f:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    receipts.append({'name': name, 'command': command, 'returncode': result.returncode})
    (RAW / 'commands.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(name + ' failed: ' + (RAW / (name + '.log')).read_text()[-2500:])


def metrics(port):
    return json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=10))


def production_identity():
    config = json.loads((RAW / 'production-resolved.json').read_text())
    command = 'pid=$(pgrep -x dgpp-serve); test "$(printf "%s\\n" "$pid" | wc -l)" -eq 1 && sha256sum /proc/$pid/exe'
    result = {}
    for node in config['nodes']:
        call = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'stephen@' + node, command], capture_output=True, text=True, timeout=20)
        assert call.returncode == 0 and call.stdout.split()[0] == EXPECTED_PROD, (node, call.stdout, call.stderr)
        result[node] = call.stdout
    return result


IMAGE = 'sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a'
NAME = 'dgpp-issue4-vllm-ref-20260923'
MODEL_HOST = '/home/stephen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4'
MODEL_CONTAINER = '/hf-model/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47'
SOURCE = '/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py'
CACHE = '/tmp/dgpp-issue4-vllm-cache-20260923'
PATCH = '/tmp/dgpp-issue4-vllm-ple-20260923.py'
EXACT = ROOT / 'benchmarks/results/2026-09-23-issue4-exact'

def on(rank, argv):
    return argv if rank == 0 else ['ssh', '-o', 'BatchMode=yes', 'stephen@192.168.88.12', shlex.join(argv)]

def inspect(rank, argv):
    return subprocess.check_output(on(rank, argv), text=True, timeout=30).strip()

for rank in (0,1):
    identity = inspect(rank, ['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE])
    assert identity == IMAGE
    assert not inspect(rank, ['docker', 'ps', '-aq', '--filter', 'name=^/' + NAME + '$'])
    run(f'cache-dir-{rank}', on(rank, ['mkdir', '-p', CACHE]))
shutil.copy2(RAW / 'ple_layer.py', PATCH)
run('copy-reference-patch', ['scp', str(RAW / 'ple_layer.py'), 'stephen@192.168.88.12:' + PATCH])
TRACE_PACKAGE = '/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/'
TOPK_HELPER = '/tmp/dgpp-issue4-vllm-topk-20260923.py'
QSA_PATCH = '/tmp/dgpp-issue4-vllm-qsa-20260923.py'
for source, target in ((RECORD/'issue4_topk.py',TOPK_HELPER),(RAW/'qsa.py',QSA_PATCH)):
    shutil.copy2(source,target)
    run('copy-'+Path(target).stem,['scp',target,'stephen@192.168.88.12:'+target])
TRACE_MODEL = '/tmp/dgpp-issue4-world-trace-model.py'
TRACE_HELPER = '/tmp/dgpp-issue4-world-trace-helper.py'
TRACE_GDN = '/tmp/dgpp-issue4-world-trace-gdn.py'
TRACE_PEER = '/tmp/dgpp-issue4-state-fp32-reference-capture-20260923'
TRACE_GDN_SOURCE = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
assert not (RAW/'captures').exists()
(RAW/'captures').mkdir()
assert not inspect(1,['sh','-c','test -e '+TRACE_PEER+' && echo exists || true'])
run('peer-capture-dir',on(1,['mkdir',TRACE_PEER]))
for source,target in ((RAW/'model.py',TRACE_MODEL),(RECORD/'issue4_trace.py',TRACE_HELPER),(RAW/'qwen_gdn_linear_attn.py',TRACE_GDN)):
    shutil.copy2(source,target)
    run('copy-'+Path(target).stem,['scp',target,'stephen@192.168.88.12:'+target])

PINNED_GDN = '/tmp/dgpp-issue4-state-fp32-configs.json'
shutil.copy2(RECORD/'pinned-gdn-configs.json',PINNED_GDN)
run('copy-pinned-configs',['scp',PINNED_GDN,'stephen@192.168.88.12:'+PINNED_GDN])
unit_summary=json.loads((RECORD/'unit-summary.json').read_text())
assert unit_summary['all_pass'] and unit_summary['alignment_sha256']==sha(RECORD/'issue4_moe_align.py')
MARLIN_OVERLAY = '/tmp/dgpp-issue4-canonical-marlin.py'
MARLIN_SOURCE = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py'
ALIGN_HELPER = '/tmp/dgpp-issue4-canonical-moe-align.py'
for source,target in ((RAW/'marlin_moe.py',MARLIN_OVERLAY),(RECORD/'issue4_moe_align.py',ALIGN_HELPER)):
    shutil.copy2(source,target)
    run('copy-'+Path(target).stem,['scp',target,'stephen@192.168.88.12:'+target])
assert sha(ROOT / 'build-release/dgpp-serve') == EXPECTED_PROD
before = metrics(18080)
assert before['scheduler']['active'] == before['scheduler']['queued'] == 0
(RAW / 'production-before.json').write_text(json.dumps({'metrics': before, 'binaries': production_identity()}, indent=2))
started = []
stopped = False
try:
    stopped = True
    run('production-down', [sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'down', '--config', str(PROD)], timeout=120)
    for rank in (1,0):
        envs = {
            'HF_HOME':'/cache/huggingface', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
            'VLLM_CACHE_ROOT':'/cache/vllm-marlin', 'TRITON_CACHE_DIR':'/cache/triton',
            'VLLM_HOST_IP':f'192.168.88.{11+rank}', 'VLLM_ENGINE_READY_TIMEOUT_S':'3600',
            'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
            'TORCH_CUDA_ARCH_LIST':'12.1a', 'FLASHINFER_CUDA_ARCH_LIST':'12.1a',
            'NCCL_NET':'IB', 'NCCL_IB_DISABLE':'0', 'NCCL_IB_HCA':'rocep1s0f0',
            'NCCL_IB_GID_INDEX':'3', 'NCCL_SOCKET_IFNAME':'enp1s0f0np0',
            'GLOO_SOCKET_IFNAME':'enp1s0f0np0', 'TP_SOCKET_IFNAME':'enp1s0f0np0',
            'NCCL_CUMEM_ENABLE':'0', 'NCCL_NVLS_ENABLE':'0',
        }
        command = ['docker','run','-d','--pull','never','--gpus','all','--name',NAME,
                   '--restart','no','--memory','110g','--memory-swap','110g',
                   '--network','host','--ipc','host','--shm-size','32g',
                   '--ulimit','memlock=-1:-1','--ulimit','stack=67108864',
                   '--cap-add','IPC_LOCK','--cap-add','SYS_NICE',
                   '--device','/dev/infiniband:/dev/infiniband',
                   '-v',MODEL_HOST+':/hf-model:ro','-v',PATCH+':'+SOURCE+':ro',
                   '-v',CACHE+':/cache',
                   '-v',QSA_PATCH+':'+TRACE_PACKAGE+'ops/qsa.py:ro',
                   '-v',TOPK_HELPER+':'+TRACE_PACKAGE+'issue4_topk.py:ro',
                   '-v',TRACE_MODEL+':'+TRACE_PACKAGE+'model.py:ro',
                   '-v',TRACE_HELPER+':'+TRACE_PACKAGE+'issue4_trace.py:ro',
                   '-v',TRACE_GDN+':'+TRACE_GDN_SOURCE+':ro',
                   '-v',MARLIN_OVERLAY+':'+MARLIN_SOURCE+':ro',
                   '-v',ALIGN_HELPER+':'+TRACE_PACKAGE+'issue4_moe_align.py:ro',
                   '-v',PINNED_GDN+':/gdn-configs.json:ro',
                   '-v',(str(RAW/'captures') if rank==0 else TRACE_PEER)+':/capture']
        for key,val in envs.items(): command += ['-e',key+'='+val]
        command += ['--entrypoint','vllm',IMAGE,'serve',MODEL_CONTAINER,
                    '--served-model-name','nvidia/Qwen3.8-Flash-Next-NVFP4',
                    '--host','127.0.0.1','--port','18085',
                    '--distributed-executor-backend','mp','--nnodes','2',
                    '--master-addr','192.168.88.11','--master-port','29641',
                    '--tensor-parallel-size','2','--dtype','bfloat16','--kv-cache-dtype','auto',
                    '--max-model-len','262144','--max-num-seqs','1','--max-num-batched-tokens','2048',
                    '--gpu-memory-utilization','0.835','--kv-cache-memory-bytes','8589934592',
                    '--enable-chunked-prefill','--no-enable-prefix-caching','--mamba-cache-mode','none',
                    '--mamba-ssm-cache-dtype','float32','--moe-backend','marlin','--enforce-eager','--reasoning-parser','qwen3','--enable-auto-tool-choice',
                    '--tool-call-parser','qwen3_coder','--node-rank',str(rank)]
        if rank: command += ['--headless']
        started.append(rank)
        run(f'reference-up-{rank}', on(rank, command), timeout=120)
    deadline = time.monotonic()+3600
    while True:
        for rank in started:
            state=json.loads(inspect(rank,['docker','inspect','--format','{{json .State}}',NAME]))
            assert state['Running'], (rank,state)
        try:
            with urllib.request.urlopen('http://127.0.0.1:18085/health',timeout=5) as r:
                if r.status==200: break
        except (OSError, urllib.error.HTTPError): pass
        if time.monotonic()>deadline: raise TimeoutError('reference startup')
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'reference starting',flush=True)
        time.sleep(15)
    original_gdn = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
    original_gdn_sha = sha(RAW/'qwen_gdn_linear_attn.py')
    source_identity = {}
    for rank in (0,1):
        logs = subprocess.check_output(on(rank,['docker','logs',NAME]),stderr=subprocess.STDOUT,text=True,timeout=30)
        (RAW/f'rank{rank}-ready.log').write_text(logs)
        assert 'MARLIN' in logs and 'Issue4 control: recurrent GDN prefill' not in logs
        assert 'Issue4: pinned GDN configurations' in logs
        digest = inspect(rank,['docker','exec',NAME,'sha256sum',original_gdn]).split()[0]
        assert digest == original_gdn_sha
        marlin_digest=inspect(rank,['docker','exec',NAME,'sha256sum',MARLIN_SOURCE]).split()[0]
        align_digest=inspect(rank,['docker','exec',NAME,'sha256sum',TRACE_PACKAGE+'issue4_moe_align.py']).split()[0]
        assert marlin_digest==sha(RAW/'marlin_moe.py') and align_digest==sha(RECORD/'issue4_moe_align.py')
        source_identity[rank]={'gdn_sha256':digest,'marlin_selected':True,'canonical_marlin_sha256':marlin_digest,'align_sha256':align_digest}
    (RAW/'runtime-source.json').write_text(json.dumps(source_identity,indent=2)+'\n')
    request=json.loads((EXACT/'raw/request.json').read_text())
    tokenize={k:request[k] for k in ('model','messages','chat_template_kwargs')}
    tokenize['add_generation_prompt']=True
    req=urllib.request.Request('http://127.0.0.1:18085/tokenize', data=json.dumps(tokenize).encode(),headers={'Content-Type':'application/json'})
    tokens=json.load(urllib.request.urlopen(req,timeout=60))
    (RAW/'tokenize.json').write_text(json.dumps(tokens))
    assert tokens['tokens']==json.loads((EXACT/'raw/current-native.ids.json').read_text())
    (RAW/'metrics-before.txt').write_bytes(urllib.request.urlopen('http://127.0.0.1:18085/metrics',timeout=10).read())
    responses=[]
    for repetition in (1,2):
        payload=(EXACT/'raw/request.json').read_bytes()
        req=urllib.request.Request('http://127.0.0.1:18085/v1/chat/completions',data=payload,headers={'Content-Type':'application/json'})
        print('Exact reference generation',repetition,'started',flush=True)
        start=time.monotonic()
        data=urllib.request.urlopen(req,timeout=3600).read()
        (RAW/f'response-{repetition}.json').write_bytes(data)
        response=json.loads(data)
        spec=importlib.util.spec_from_file_location('issue4_replay',EXACT/'replay.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        verdict=module.score(response,json.loads((EXACT/'raw/oracle.json').read_text()))
        verdict['raw_protocol_pass']=verdict['protocol_pass']
        verdict['protocol_pass']=verdict['finish_reason']=='stop' and verdict['usage']['prompt_tokens']==261120
        verdict['wall_seconds']=time.monotonic()-start
        verdict['native_tokens_exact']=True
        verdict['prefix_cache_disabled']=True
        (RAW/f'verdict-{repetition}.json').write_text(json.dumps(verdict,indent=2)+'\n')
        responses.append(response['choices'][0]['message']['content'])
        print('Reference verdict',repetition,verdict,flush=True)
        (RAW/f'metrics-after-{repetition}.txt').write_bytes(urllib.request.urlopen('http://127.0.0.1:18085/metrics',timeout=10).read())

    first=json.loads((RAW/'response-1.json').read_text())
    second=json.loads((RAW/'response-2.json').read_text())
    (RAW/'repeatability.json').write_text(json.dumps({'same_assistant_text':responses[0]==responses[1],
        'same_usage':first['usage']==second['usage']},indent=2)+'\n')
    time.sleep(2)

finally:
    cleanup_errors = []
    for rank in reversed(started):
        try:
            exists=inspect(rank,['docker','ps','-aq','--filter','name=^/'+NAME+'$'])
            if not exists: continue
            with (RAW/f'rank{rank}.log').open('w') as f:
                subprocess.run(on(rank,['docker','logs',NAME]),stdout=f,stderr=subprocess.STDOUT,timeout=30)
            subprocess.run(on(rank,['docker','stop','-t','30',NAME]),stdout=subprocess.DEVNULL,timeout=60)
            state=json.loads(inspect(rank,['docker','inspect','--format','{{json .State}}',NAME]))
            (RAW/f'rank{rank}-state.json').write_text(json.dumps(state,indent=2)+'\n')
            assert not state['Running']
            subprocess.run(on(rank,['docker','rm',NAME]),check=True,stdout=subprocess.DEVNULL,timeout=30)
        except Exception as error:
            cleanup_errors.append(str(error))
    if cleanup_errors:
        raise RuntimeError('Reference cleanup needs attention before restoring production: '+str(cleanup_errors))
    if stopped:
        run('production-restore',[sys.executable,str(ROOT/'scripts/dgpp-cluster'),'up','--config',str(PROD),'--bin',str(ROOT/'build-release/dgpp-serve')])
        binaries=production_identity()
        resolved=json.loads(Path('/home/stephen/dgpp/log/deployments/a911a2d7c4b3be6a/cluster.resolved.json').read_text())
        assert resolved==json.loads((RAW/'production-resolved.json').read_text())
        request={'model':resolved['model'],'messages':[{'role':'user','content':'Reply exactly OK.'}],'temperature':0,'max_tokens':128,'reasoning_effort':'minimal'}
        response=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'}),timeout=90))
        assert response['choices'][0]['message']['content'].strip()=='OK'
        m=metrics(18080);assert m['scheduler']['active']==m['scheduler']['queued']==0 and not m['service']['engine_failed']
        (RAW/'restoration.json').write_text(json.dumps({'binaries':binaries,'config_matches':True,'smoke':response,'idle':True},indent=2)+'\n')
        print('Production restored and independently verified',flush=True)

print('W4A16 reference control complete',flush=True)

with (RAW/'copy-peer-captures.log').open('w') as copy_log:
    peer=subprocess.Popen(on(1,['tar','-C',TRACE_PEER,'-cf','-','rank1']),stdout=subprocess.PIPE,stderr=copy_log)
    local=subprocess.Popen(['tar','-C',str(RAW/'captures'),'-xf','-'],stdin=peer.stdout,stdout=copy_log,stderr=copy_log)
    peer.stdout.close()
    local_result=local.wait(timeout=600)
    peer_result=peer.wait(timeout=60)
    assert local_result==peer_result==0,(local_result,peer_result)
print('Peer trace collected',flush=True)
