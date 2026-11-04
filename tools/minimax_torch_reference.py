#!/usr/bin/env python3
"""Independent, layer-streamed MiniMax text reference using PyTorch operations.

This is an accuracy instrument, not a serving implementation or benchmark.
It decodes the published weights without activation quantization, accumulates
projections in FP32, and uses FP32 attention probabilities. Simulated TP keeps
local BF16 projection/fold boundaries explicit. No dgpp kernels are loaded.
Only the safetensors reader and format tables are shared with the NumPy oracle.

Each manifest entry supplies a name and text or explicit ids. Logs use the
strict prefill scoring format accepted by scripts/fabric_logprob.py --prefill.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from minimax_reference_dump import FP4, FP8, Weights


class Reference:
    def __init__(self, checkpoint, world, rounded_residual):
        self.w = Weights(checkpoint)
        self.c = json.loads((checkpoint / 'config.json').read_text())['text_config']
        self.world = world
        self.rounded_residual = rounded_residual
        self.config_sha256 = hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest()
        self.checkpoint = str(checkpoint.resolve())
        self.fp8 = torch.tensor(FP8, device='cuda')
        self.fp4 = torch.tensor(FP4, device='cuda')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def tensor(self, name, rows=None):
        raw = self.w.raw(name, rows)
        t = torch.from_numpy(np.array(raw, copy=True)).cuda()
        dtype = self.w.header[name]['dtype']
        if dtype == 'BF16':
            return t.view(torch.bfloat16)
        if dtype == 'F8_E4M3':
            return self.fp8[t.long()]
        return t

    def matrix(self, base):
        if base+'.weight' in self.w.header:
            weight = self.tensor(base+'.weight')
            if base+'.weight_scale_inv' in self.w.header:
                scale = torch.exp2(self.tensor(base+'.weight_scale_inv').float()-127)
                weight *= scale.repeat_interleave(32, dim=1)
            return weight.to(torch.bfloat16), 1.0
        codes = self.tensor(base+'.weight_packed')
        codes = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(1)
        weight = self.fp4[codes.long()]
        weight *= self.tensor(base+'.weight_scale').repeat_interleave(16, dim=1)
        # FP4 * FP8 has at most six significant bits: BF16 holds it exactly.
        # Keep the FP32 global divisor outside the dot, before output rounding.
        global_scale = float(self.w.tensor(base+'.weight_global_scale'))
        return weight.to(torch.bfloat16), global_scale

    @staticmethod
    def mm(x, weight, divisor=1.0, rounded=True):
        y = torch.mm(x, weight.T, out_dtype=torch.float32) / divisor
        return y.to(torch.bfloat16) if rounded else y

    def linear(self, x, base):
        return self.mm(x, *self.matrix(base))

    def parts(self, x, base, rounded=True):
        weight, divisor = self.matrix(base)
        width = weight.shape[1] // self.world
        return [self.mm(x[:, r*width:(r+1)*width].contiguous(),
                        weight[:, r*width:(r+1)*width].contiguous(), divisor, rounded)
                for r in range(self.world)]

    @staticmethod
    def fold(parts):
        y = torch.zeros_like(parts[0], dtype=torch.float32)
        for p in parts:
            y += p.float()
        return y.to(torch.bfloat16)

    def norm(self, x, base):
        x = x.float()
        inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True)+self.c['rms_norm_eps'])
        return (x*inv*(1+self.tensor(base+'.weight').float())).to(torch.bfloat16)

    def add_norm(self, x, add, base):
        value = x.float()+add.float()
        residual = value.to(torch.bfloat16)
        # The published fused Gemma norm uses the unrounded sum. The explicit
        # diagnostic option isolates dgpp's existing rounded-residual recipe.
        return residual, self.norm(residual if self.rounded_residual else value, base)

    def rotary(self, x):
        d = self.c['rotary_dim']
        inv = (1 / torch.pow(torch.tensor(self.c['rope_theta'], device='cuda', dtype=torch.float64),
                            torch.arange(0,d,2,device='cuda',dtype=torch.float64)/d)).float()
        angle = torch.arange(x.shape[0],device='cuda',dtype=torch.float32)[:,None]*inv[None,:]
        cs, sn = angle.cos()[:,None,:], angle.sin()[:,None,:]
        lo, hi = x[...,:d//2].float(), x[...,d//2:d].float()
        return torch.cat((lo*cs-hi*sn, hi*cs+lo*sn, x[...,d:].float()),dim=-1).to(torch.bfloat16)

    def attention(self, n, layer):
        c, t = self.c, n.shape[0]
        qh, kh, d = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        ih = c['sparse_attention_config']['sparse_num_index_heads']
        topk = c['sparse_attention_config']['sparse_topk_blocks']
        p = f'language_model.model.layers.{layer}.self_attn.'
        def proj(name, heads, normalized=True):
            y = self.linear(n,p+name+'_proj').reshape(t,heads,d)
            return self.rotary(self.norm(y,p+name+'_norm')) if normalized else y
        q,k,v = proj('q',qh),proj('k',kh),proj('v',kh,False)
        iq,ik = proj('index_q',ih),proj('index_k',1)
        out = torch.empty_like(q)
        blocks = (t+127)//128
        positions = torch.arange(t,device='cuda')
        offsets = torch.arange(128,device='cuda')
        for at in range(0,t,32):
            end = min(at+32,t)
            rowpos = positions[at:end]
            for index in range(ih):
                index_scores = torch.mm(iq[at:end,index].float(),ik[:,0].float().T)
                index_scores.masked_fill_(positions[None,:]>rowpos[:,None], -torch.inf)
                index_scores = torch.nn.functional.pad(index_scores,(0,blocks*128-t),value=-torch.inf)
                maximum = index_scores.view(end-at,blocks,128).amax(-1)
                maximum.scatter_(1,(rowpos//128)[:,None],torch.inf)
                # Stable tie breaking prefers the smaller block id, then sort
                # selected blocks into token order for the reference reduction.
                selected = maximum.argsort(dim=1,descending=True,stable=True)[:,:min(topk,blocks)]
                selected = selected.sort(dim=1).values
                tokens = (selected[:,:,None]*128+offsets).flatten(1)
                mask = tokens<=rowpos[:,None]
                safe = tokens.clamp(max=t-1)
                first,last = index*(qh//ih),(index+1)*(qh//ih)
                kv = first//(qh//kh)
                keys,values = k[safe,kv].float(),v[safe,kv].float()
                scores = torch.bmm(q[at:end,first:last].float(),keys.transpose(1,2)) / math.sqrt(d)
                scores.masked_fill_(~mask[:,None,:],-torch.inf)
                out[at:end,first:last] = torch.bmm(scores.softmax(-1),values).to(torch.bfloat16)
        return self.fold(self.parts(out.flatten(1),p+'o_proj'))

    def mlp(self, n, base, expert=False):
        c = self.c
        gate,up,down = ('w1','w3','w2') if expert else ('gate_proj','up_proj','down_proj')
        g = self.linear(n,base+gate).float().clamp(max=c['swiglu_limit'])
        u = self.linear(n,base+up).float().clamp(-c['swiglu_limit'],c['swiglu_limit'])
        act = (g*torch.sigmoid(c['swiglu_alpha']*g)*(u+1)).to(torch.bfloat16)
        return self.parts(act,base+down,rounded=not expert)

    def ffn(self, n, layer):
        c = self.c
        p = f'language_model.model.layers.{layer}.'
        if not c['moe_layer_freq'][layer]:
            return self.fold(self.mlp(n,p+'mlp.'))
        p += 'block_sparse_moe.'
        scores = torch.sigmoid(torch.mm(n.float(),self.tensor(p+'gate.weight').T))
        biased = scores+self.tensor(p+'e_score_correction_bias')
        ids = biased.argsort(dim=-1,descending=True,stable=True)[:,:c['num_experts_per_tok']].sort(dim=-1).values
        picked = scores.gather(1,ids)
        weights = picked/picked.sum(dim=1,keepdim=True).clamp(min=1e-20)*c['routed_scaling_factor']
        routed = [torch.zeros_like(n,dtype=torch.float32) for _ in range(self.world)]
        for e in range(c['num_local_experts']):
            rows,slots = torch.where(ids==e)
            if rows.numel():
                parts = self.mlp(n[rows],p+f'experts.{e}.',expert=True)
                for r,y in enumerate(parts):
                    routed[r][rows] += weights[rows,slots,None]*y
        shared = self.mlp(n,p+'shared_experts.')
        return self.fold([(r.to(torch.bfloat16).float()+s.float()).to(torch.bfloat16)
                          for r,s in zip(routed,shared)])

    def run(self, cases, output):
        inputs = [self.tensor('language_model.model.embed_tokens.weight', np.array(case['ids'][:-1])) for case in cases]
        states = [self.norm(x,'language_model.model.layers.0.input_layernorm') for x in inputs]
        for layer in range(self.c['num_hidden_layers']):
            start = time.monotonic()
            norms = []
            for case_index,(x,n) in enumerate(zip(inputs,states)):
                p = f'language_model.model.layers.{layer}.'
                a = self.attention(n,layer)
                x,n = self.add_norm(x,a,p+'post_attention_layernorm')
                inputs[case_index] = x
                norms.append(n)
            outputs = self.ffn(torch.cat(norms),layer).split([len(n) for n in norms])
            for case_index,(x,y) in enumerate(zip(inputs,outputs)):
                next_norm = f'language_model.model.layers.{layer+1}.input_layernorm' if layer+1<self.c['num_hidden_layers'] else 'language_model.model.norm'
                inputs[case_index],states[case_index] = self.add_norm(x,y,next_norm)
            torch.cuda.synchronize()
            print(f'layer {layer} complete in {time.monotonic()-start:.2f}s',flush=True)
        # Read vocabulary slabs so even the output head fits a small reference GPU.
        vocab = self.c['vocab_size']
        for case,hidden in zip(cases,states):
            folder = output/case['name']
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'metadata.json').write_text(json.dumps(dict(world=self.world,ids=case['ids'],rounded_residual=self.rounded_residual,
                backend='pytorch independent FP32 attention', torch_version=torch.__version__,
                checkpoint=self.checkpoint,config_sha256=self.config_sha256),indent=2)+'\n')
            for rank in range(self.world):
                begin,end = rank*vocab//self.world,(rank+1)*vocab//self.world
                weight = self.tensor('language_model.lm_head.weight',slice(begin,end))
                with (folder/f'r{rank}.log').open('w') as log:
                    for at in range(0,len(hidden),64):
                        y = self.mm(hidden[at:at+64],weight,rounded=False).double()
                        lse = y.logsumexp(-1).cpu().tolist()
                        maximum,index = y.max(-1)
                        maximum,index = maximum.cpu().tolist(),index.cpu().tolist()
                        for row in range(len(y)):
                            step = at+row
                            target = case['ids'][step+1]
                            target_logit = y[row,target-begin].item() if begin<=target<end else math.nan
                            log.write(f'[prefill_tf] rank {rank} step {step}: target {target} argmax {begin+index[row]} lmax {maximum[row]:.9g} lse {lse[row]:.12g} target_logit {target_logit:.9g}\n')
                    log.write(f'[prefill_tf_end] rank {rank} world {self.world} positions {len(hidden)}\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--world',type=int,choices=[1,2,4],default=4)
    p.add_argument('--rounded-residual',action='store_true',help='diagnostic: normalize the rounded BF16 residual sum')
    args = p.parse_args()
    cases = json.loads(args.manifest.read_text())
    tok = Tokenizer.from_file(str(args.checkpoint_dir/'tokenizer.json'))
    for case in cases:
        if 'ids' not in case:
            case['ids'] = tok.encode(case['text'],add_special_tokens=False).ids
        if len(case['ids'])<2 or Path(case['name']).name!=case['name']:
            p.error('each case needs a simple name and at least two tokens')
        print(f"{case['name']}: {len(case['ids'])} tokens",flush=True)
    with torch.inference_mode():
        Reference(args.checkpoint_dir,args.world,args.rounded_residual).run(cases,args.output)


if __name__=='__main__':
    main()
