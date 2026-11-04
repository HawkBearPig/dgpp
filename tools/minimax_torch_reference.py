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
    def __init__(self, checkpoint, world, rounded_residual, projection_rows=0, projection_backend='cublas',
                 elementwise_backend='torch', attention_backend='torch'):
        self.w = Weights(checkpoint)
        self.c = json.loads((checkpoint / 'config.json').read_text())['text_config']
        self.world = world
        self.rounded_residual = rounded_residual
        self.projection_rows = projection_rows
        self.projection_backend = projection_backend
        self.elementwise_backend = elementwise_backend
        self.attention_backend = attention_backend
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

    def mm(self, x, weight, divisor=1.0, rounded=True, streaming_order=False):
        # A reference-only control: changing cuBLAS's row geometry while
        # preserving every BF16 boundary measures its own reduction sensitivity.
        rows = self.projection_rows or len(x)
        if self.projection_backend == 'triton-fixed':
            from minimax_reference_gemm import projection
            parts = [projection(part,weight,streaming_order) for part in x.split(rows)]
        else:
            parts = [torch.mm(part, weight.T, out_dtype=torch.float32) for part in x.split(rows)]
        y = torch.cat(parts)
        if self.elementwise_backend == 'triton-fixed':
            from minimax_reference_arithmetic import divide
            y = divide(y, divisor) if divisor != 1.0 else y
        else:
            y = y / divisor
        return y.to(torch.bfloat16) if rounded else y

    def linear(self, x, base):
        return self.mm(x, *self.matrix(base), streaming_order=base+'.weight_scale_inv' in self.w.header)

    def parts(self, x, base, rounded=True):
        weight, divisor = self.matrix(base)
        width = weight.shape[1] // self.world
        return [self.mm(x[:, r*width:(r+1)*width].contiguous(),
                        weight[:, r*width:(r+1)*width].contiguous(), divisor, rounded,
                        streaming_order=base+'.weight_scale_inv' in self.w.header)
                for r in range(self.world)]

    @staticmethod
    def fold(parts):
        y = torch.zeros_like(parts[0], dtype=torch.float32)
        for p in parts:
            y += p.float()
        return y.to(torch.bfloat16)

    def norm(self, x, base):
        if self.elementwise_backend == 'triton-fixed':
            from minimax_reference_arithmetic import norm
            return norm(x, self.tensor(base+'.weight'), self.c['rms_norm_eps'])
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
        # Published CUDA recipe: FP32 exponent, power and reciprocal. A
        # double intermediate changes 14/32 frequencies at theta=1e7 and
        # causes material phase drift near the million-token limit.
        inv = 1 / torch.pow(self.c['rope_theta'],
                            torch.arange(0,d,2,device='cuda',dtype=torch.float32)/d)
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
        if self.attention_backend == 'triton-fixed':
            from minimax_reference_attention import attention, select_blocks
            positions = torch.arange(t, device='cuda', dtype=torch.int64)
            selected = select_blocks(iq, ik, positions, topk)
            out = attention(q, k, v, selected, positions)
            return self.fold(self.parts(out.flatten(1),p+'o_proj'))
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
        if self.elementwise_backend == 'triton-fixed':
            from minimax_reference_arithmetic import activation
            act = activation(g, u, c['swiglu_alpha'], c['swiglu_limit'])
        else:
            act = (g*torch.sigmoid(c['swiglu_alpha']*g)*(u+1)).to(torch.bfloat16)
        return self.parts(act,base+down,rounded=not expert)

    def ffn(self, n, layer):
        c = self.c
        p = f'language_model.model.layers.{layer}.'
        if not c['moe_layer_freq'][layer]:
            self.last_routes = None
            return self.fold(self.mlp(n,p+'mlp.'))
        p += 'block_sparse_moe.'
        if self.elementwise_backend == 'triton-fixed':
            from minimax_reference_arithmetic import router
            scores, _ = router(n, self.tensor(p+'gate.weight'))
        else:
            scores = torch.sigmoid(torch.mm(n.float(),self.tensor(p+'gate.weight').T))
        biased = scores+self.tensor(p+'e_score_correction_bias')
        ids = biased.argsort(dim=-1,descending=True,stable=True)[:,:c['num_experts_per_tok']].sort(dim=-1).values
        self.last_routes = ids
        picked = scores.gather(1,ids)
        if self.elementwise_backend == 'triton-fixed':
            denominator = torch.zeros_like(picked[:, :1])
            for slot in range(picked.shape[1]):
                denominator += picked[:, slot:slot+1]
        else:
            denominator = picked.sum(dim=1,keepdim=True)
        weights = picked/denominator.clamp(min=1e-20)*c['routed_scaling_factor']
        routed = [torch.zeros_like(n,dtype=torch.float32) for _ in range(self.world)]
        for e in range(c['num_local_experts']):
            rows,slots = torch.where(ids==e)
            if rows.numel():
                parts = self.mlp(n[rows],p+f'experts.{e}.',expert=True)
                for r,y in enumerate(parts):
                    if self.elementwise_backend == 'triton-fixed':
                        from minimax_reference_arithmetic import weighted_add
                        weighted_add(routed[r], y, weights[rows, slots], rows)
                    else:
                        routed[r][rows] += weights[rows,slots,None]*y
        shared = self.mlp(n,p+'shared_experts.')
        return self.fold([(r.to(torch.bfloat16).float()+s.float()).to(torch.bfloat16)
                          for r,s in zip(routed,shared)])

    def check_stages(self, engine, output, layers=0):
        """Check each operation with recorded inputs, separately from trajectory quality."""
        total_layers = self.c['num_hidden_layers']
        if not 0 <= layers <= total_layers:
            raise ValueError('diagnostic layer count outside the model')
        layers = layers or total_layers
        metadata = json.loads((engine/'metadata.json').read_text())
        if metadata.get('world', 1) != self.world:
            raise ValueError('engine/reference TP worlds differ')
        if metadata.get('residual_norm', 'bf16') != ('bf16' if self.rounded_residual else 'fp32'):
            raise ValueError('engine/reference residual-normalization recipes differ')
        if metadata.get('rope_inv_freq') != 'cuda-fp32':
            raise ValueError('engine/reference rotary frequency recipes differ; collect a new trace')
        rows, hidden = len(metadata['tokens']), self.c['hidden_size']
        metrics, failures = [], []

        def read(stage, layer):
            data = np.fromfile(engine/f'{stage}-{layer}.bf16', dtype='<u2').reshape(rows, hidden)
            return torch.from_numpy(data).cuda().view(torch.bfloat16)

        def compare(stage, layer, expected):
            actual = read(stage, layer)
            error = (actual.double()-expected.double())
            rms = (error.norm()/expected.double().norm().clamp(min=1e-20)).item()
            per_row = error.norm(dim=1)/expected.double().norm(dim=1).clamp(min=1e-20)
            metrics.append(dict(layer=layer, stage=stage, rms=rms,
                                changed_values=int(torch.count_nonzero(actual != expected)),
                                max_row_rms=per_row.max().item(), worst_row=per_row.argmax().item()))
            print(f'Layer {layer} {stage}: relative RMS {rms:.6g}', flush=True)
            if not math.isfinite(rms) or rms > .01:
                failures.append(f'layer {layer} {stage}: relative RMS {rms} exceeds 1%')
            return actual

        x = self.tensor('language_model.model.embed_tokens.weight', np.array(metadata['tokens']))
        for layer in range(layers):
            p = f'language_model.model.layers.{layer}.'
            n = compare('input_norm', layer, self.norm(x, p+'input_layernorm'))
            a = compare('attention', layer, self.attention(n, layer))
            x, expected = self.add_norm(x.to(torch.bfloat16), a, p+'post_attention_layernorm')
            n = compare('ffn_norm', layer, expected)
            y = compare('ffn', layer, self.ffn(n, layer))
            if self.last_routes is not None:
                actual = np.fromfile(engine/f'routes-{layer}.i32', dtype='<i4').reshape(rows, -1)
                changed = np.flatnonzero(np.any(actual != self.last_routes.cpu().numpy(), axis=1))
                print(f'Layer {layer} router: {len(changed)} mismatched rows', flush=True)
                if len(changed):
                    failures.append(f'layer {layer}: router mismatch at rows {changed.tolist()}')
            x = x.float()+y.float()
            saved = read('layer', layer)
            if not torch.equal(saved, x.to(torch.bfloat16)):
                failures.append(f'layer {layer}: residual composition differs')
            if self.rounded_residual:
                x = saved
        output.mkdir(parents=True, exist_ok=True)
        (output/'stages.json').write_text(json.dumps(dict(metrics=metrics, failures=failures,
            layers_checked=layers,layers_total=total_layers,complete=layers==total_layers,
            projection_backend=self.projection_backend,elementwise_backend=self.elementwise_backend,
            attention_backend=self.attention_backend), indent=2)+'\n')
        if failures:
            raise AssertionError('\n'.join(failures))
        print(f'Isolated sublayer checks passed for {layers}/{total_layers} layers; this is not a full-trajectory quality verdict.', flush=True)

    def run(self, cases, output, dump_states=False):
        def record(case_index, stage, layer, tensor):
            if not dump_states:
                return
            folder = output/cases[case_index]['name']
            folder.mkdir(parents=True,exist_ok=True)
            (folder/f'{stage}-{layer}.bf16').write_bytes(tensor.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes())
        inputs = [self.tensor('language_model.model.embed_tokens.weight', np.array(case['ids'][:-1])) for case in cases]
        states = [self.norm(x,'language_model.model.layers.0.input_layernorm') for x in inputs]
        for layer in range(self.c['num_hidden_layers']):
            start = time.monotonic()
            norms = []
            for case_index,(x,n) in enumerate(zip(inputs,states)):
                p = f'language_model.model.layers.{layer}.'
                record(case_index,'input_norm',layer,n)
                a = self.attention(n,layer)
                record(case_index,'attention',layer,a)
                x,n = self.add_norm(x,a,p+'post_attention_layernorm')
                record(case_index,'ffn_norm',layer,n)
                inputs[case_index] = x
                norms.append(n)
            outputs = self.ffn(torch.cat(norms),layer).split([len(n) for n in norms])
            routes = self.last_routes.split([len(n) for n in norms]) if self.last_routes is not None else None
            for case_index,(x,y) in enumerate(zip(inputs,outputs)):
                record(case_index,'ffn',layer,y)
                next_norm = f'language_model.model.layers.{layer+1}.input_layernorm' if layer+1<self.c['num_hidden_layers'] else 'language_model.model.norm'
                inputs[case_index],states[case_index] = self.add_norm(x,y,next_norm)
                record(case_index,'layer',layer,inputs[case_index])
                if dump_states and routes is not None:
                    (output/cases[case_index]['name']/f'routes-{layer}.i32').write_bytes(routes[case_index].to(torch.int32).cpu().numpy().tobytes())
            torch.cuda.synchronize()
            print(f'layer {layer} complete in {time.monotonic()-start:.2f}s',flush=True)
        # Read vocabulary slabs so even the output head fits a small reference GPU.
        vocab = self.c['vocab_size']
        for case,hidden in zip(cases,states):
            score_from = case.get('score_from',1)
            input_hash = 1469598103934665603
            for token in case['ids']:
                input_hash = ((input_hash^token)*1099511628211)&((1<<64)-1)
            folder = output/case['name']
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'metadata.json').write_text(json.dumps(dict(world=self.world,ids=case['ids'],rounded_residual=self.rounded_residual,
                projection_rows=self.projection_rows,
                projection_backend=self.projection_backend,elementwise_backend=self.elementwise_backend,
                attention_backend=self.attention_backend,
                rope_inv_freq='cuda-fp32',
                score_from=score_from,
                backend='pytorch independent FP32 attention', torch_version=torch.__version__,
                checkpoint=self.checkpoint,config_sha256=self.config_sha256),indent=2)+'\n')
            for rank in range(self.world):
                begin,end = rank*vocab//self.world,(rank+1)*vocab//self.world
                weight = self.tensor('language_model.lm_head.weight',slice(begin,end))
                with (folder/f'r{rank}.log').open('w') as log:
                    log.write(f"[minimax_protocol] world {self.world} tokens {len(case['ids'])} score_from {score_from} input {input_hash:016x}\n")
                    for at in range(0,len(hidden),64):
                        y = self.mm(hidden[at:at+64],weight,rounded=False).double()
                        lse = y.logsumexp(-1).cpu().tolist()
                        maximum,index = y.max(-1)
                        maximum,index = maximum.cpu().tolist(),index.cpu().tolist()
                        for row in range(len(y)):
                            step = at+row
                            if step+1 < score_from:
                                continue
                            target = case['ids'][step+1]
                            target_logit = y[row,target-begin].item() if begin<=target<end else math.nan
                            log.write(f'[prefill_tf] rank {rank} step {step+1-score_from}: target {target} argmax {begin+index[row]} lmax {maximum[row]:.9g} lse {lse[row]:.12g} target_logit {target_logit:.9g}\n')
                    log.write(f'[prefill_tf_end] rank {rank} world {self.world} positions {len(hidden)+1-score_from}\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest',type=Path)
    mode.add_argument('--check-engine-dir',type=Path,help='check recorded sublayers using their actual inputs')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--world',type=int,choices=[1,2,4],default=4)
    p.add_argument('--rounded-residual',action='store_true',help='diagnostic: normalize the rounded BF16 residual sum')
    p.add_argument('--dump-states',action='store_true',help='save sublayer boundaries for discrepancy tracing')
    p.add_argument('--projection-rows',type=int,default=0,help='reference reduction control: cap rows per cuBLAS projection (0: full)')
    p.add_argument('--projection-backend',choices=['cublas','triton-fixed'],default='cublas',help='fixed 16-term tensor-core reductions or cuBLAS algorithm dispatch')
    p.add_argument('--elementwise-backend',choices=['torch','triton-fixed'],default='torch',help='diagnostic: fixed FP32 norm/router reductions and activation arithmetic')
    p.add_argument('--attention-backend',choices=['torch','triton-fixed'],default='torch',help='diagnostic: fixed scalar QK/index dots, online softmax and FP32 PV accumulation')
    p.add_argument('--check-layers',type=int,default=0,help='diagnostic: check only the first N recorded layers (0: all); never produces trajectory scores')
    args = p.parse_args()
    if args.projection_rows < 0:
        p.error('--projection-rows must be nonnegative')
    if args.check_layers < 0 or (args.check_layers and not args.check_engine_dir):
        p.error('--check-layers must be nonnegative and requires --check-engine-dir')
    reference = Reference(args.checkpoint_dir,args.world,args.rounded_residual,args.projection_rows,args.projection_backend,args.elementwise_backend,args.attention_backend)
    if args.check_engine_dir:
        with torch.inference_mode():
            reference.check_stages(args.check_engine_dir,args.output,args.check_layers)
        return
    cases = json.loads(args.manifest.read_text())
    tok = None
    for case in cases:
        if 'ids' not in case:
            if tok is None:
                tok = Tokenizer.from_file(str(args.checkpoint_dir/'tokenizer.json'))
            case['ids'] = tok.encode(case['text'],add_special_tokens=False).ids
        if len(case['ids'])<2 or Path(case['name']).name!=case['name']:
            p.error('each case needs a simple name and at least two tokens')
        score_from = case.get('score_from',1)
        if type(score_from) is not int or not 1 <= score_from < len(case['ids']):
            p.error('score_from must select a target token after the first input token')
        print(f"{case['name']}: {len(case['ids'])} tokens",flush=True)
    with torch.inference_mode():
        reference.run(cases,args.output,args.dump_states)


if __name__=='__main__':
    main()
