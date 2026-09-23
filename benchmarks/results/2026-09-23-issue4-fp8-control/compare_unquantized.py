#!/usr/bin/env python3
"""Compare shared non-expert BF16/F32 tensors between cached checkpoints."""
from pathlib import Path
import hashlib,json,struct
root=Path('/home/stephen/.cache/huggingface/hub')
paths=[root/'models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47',root/'models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13']
indices=[json.loads((p/'model.safetensors.index.json').read_text())['weight_map'] for p in paths]
headers={}
def info(rank,name):
 file=paths[rank]/indices[rank][name]
 if file not in headers:
  with file.open('rb') as f:
   n,=struct.unpack('<Q',f.read(8));headers[file]=(8+n,json.loads(f.read(n)))
 offset,header=headers[file];return file,offset,header[name]
def digest(record):
 file,offset,meta=record;lo,hi=meta['data_offsets'];h=hashlib.sha256()
 with file.open('rb') as f:
  f.seek(offset+lo);remaining=hi-lo
  while remaining:
   data=f.read(min(8*1024*1024,remaining));assert data;h.update(data);remaining-=len(data)
 return h.hexdigest()
results=[]
for name in sorted(set(indices[0])&set(indices[1])):
 if '.experts.' in name or not name.startswith(('model.language_model.','mtp.')):continue
 a,b=info(0,name),info(1,name)
 if a[2]['dtype'] not in ('BF16','F32') or b[2]['dtype'] not in ('BF16','F32'):continue
 ah,bh=digest(a),digest(b)
 results.append({'name':name,'dtype':a[2]['dtype'],'bytes':a[2]['data_offsets'][1]-a[2]['data_offsets'][0],'same_shape_dtype':a[2]['shape']==b[2]['shape'] and a[2]['dtype']==b[2]['dtype'],'nvfp4_checkpoint_sha256':ah,'fp8_checkpoint_sha256':bh,'matches':ah==bh})
summary={'tensors':len(results),'bytes_per_checkpoint':sum(r['bytes'] for r in results),'all_match':all(r['matches'] and r['same_shape_dtype'] for r in results),'scope':'common non-expert language-model and MTP BF16/F32 tensors; excludes quantized PLE embedding table and vision tensors'}
out=Path(__file__).resolve().parent/'unquantized-parity.json';out.write_text(json.dumps({'summary':summary,'tensors':results},indent=2)+'\n');print(summary)
