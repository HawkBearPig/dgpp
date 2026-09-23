"""CPU Torch oracle following pinned vLLM NVFP4 emulation arithmetic."""
from pathlib import Path
import json
import numpy as np
import torch
root=Path(__file__).resolve().parent
raw=root/'raw/quant';raw.mkdir(exist_ok=True)
torch.set_num_threads(2)
rng=np.random.default_rng(4)
x=torch.from_numpy((rng.normal(size=(128,2560))*np.exp2(rng.uniform(-15,10,size=(128,1)))).astype(np.float32)).to(torch.bfloat16)
x[0]=0
# Frozen real model inputs alongside a broad range of synthetic groups.
p=root.parent/'2026-09-23-issue4-operators/raw/captures/rank0/layer0/attention_input.bin'
x[64:]=torch.from_numpy(np.fromfile(p,dtype=np.uint16,count=64*2560).reshape(64,2560)).view(torch.bfloat16)
x.view(torch.uint16).numpy().tofile(raw/'input.bin')
scales=np.asarray(json.loads((root/'input-scales.json').read_text()),dtype=np.float32).reshape(-1)
inv=np.float32(1)/scales
inv.tofile(raw/'global.bin')
xf=x.float().reshape(-1,16)
amax=xf.abs().amax(dim=-1,keepdim=True)
with (raw/'expected.bin').open('wb') as out:
 for global_scale in inv:
  glob=torch.tensor(global_scale.item(),dtype=torch.float32)
  scale=(glob*(amax*torch.tensor(1/6,dtype=torch.float32))).clamp(-448,448).to(torch.float8_e4m3fn).float()
  encode=torch.where(scale==0,torch.zeros_like(scale),glob/scale)
  z=(xf*encode).clamp(-6,6); a=z.abs()
  y=torch.zeros_like(a)
  for mask,val in ((a>5,6),((a>=3.5)&(a<=5),4),((a>2.5)&(a<3.5),3),((a>=1.75)&(a<=2.5),2),((a>1.25)&(a<1.75),1.5),((a>=.75)&(a<=1.25),1),((a>.25)&(a<.75),.5)):
   y[mask]=val
  y*=torch.where(z<0,-1,1)
  result=(y*scale).to(torch.bfloat16)
  out.write(result.view(torch.uint16).numpy().tobytes())
(raw/'meta.json').write_text(json.dumps({'rows':128,'width':2560,'cases':len(inv),'scalars':128*2560*len(inv),'real_input_rows':64},indent=2)+'\n')
print('Prepared',len(inv),'cases',flush=True)
