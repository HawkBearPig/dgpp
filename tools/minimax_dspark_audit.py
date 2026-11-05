#!/usr/bin/env python3
"""Compare captured real DSpark blocks with independent native-precision arithmetic.

Uses the checkpoint's weights, pinned upstream activation producers, and PyTorch
products. This is a diagnostic, not a full upstream serving implementation.
Target verification labels are valid only along the captured proposal prefix.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from minimax_native_reference import NativeReference
from minimax_reference_dump import Weights

PREFIX = 'language_model.model.dspark.'


def load(path, dtype, shape):
    value = torch.from_numpy(np.fromfile(path, dtype=dtype).reshape(shape).copy()).cuda()
    return value.view(torch.bfloat16) if dtype == np.uint16 else value


def error(actual, reference):
    a, b = actual.float(), reference.float()
    return float(((a-b).square().mean() / b.square().mean().clamp_min(1e-30)).sqrt())


class DraftReference(NativeReference):
    @lru_cache(maxsize=None)
    def tensor(self, name, rows=None):
        return super().tensor(name, rows)

    @lru_cache(maxsize=None)
    def matrix(self, base):
        return super().matrix(base)

    def rotate(self, x, start):
        d = self.c['rotary_dim']
        inv = 1 / torch.pow(self.c['rope_theta'], torch.arange(0, d, 2, device='cuda', dtype=torch.float32)/d)
        angle = torch.arange(start, start+len(x), device='cuda', dtype=torch.float32)[:, None] * inv[None, :]
        cs, sn = angle.cos()[:, None, :], angle.sin()[:, None, :]
        lo, hi = x[..., :d//2].float(), x[..., d//2:d].float()
        return torch.cat((lo*cs-hi*sn, hi*cs+lo*sn, x[..., d:].float()), -1).to(torch.bfloat16)

    def qk(self, x, base, start, heads):
        return self.rotate(self.norm(x.reshape(len(x), heads, self.c['head_dim']), base), start)

    def block(self, embed, metadata, cache):
        c = self.c
        position, rows = metadata['position'], c['dflash_config']['block_size']
        qh, kh, dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        ids = [metadata['anchor']] + [c['dspark_noise_token_id']] * (rows-1)
        residual, pending = embed[ids], None
        kv_errors = []
        for layer in range(c['num_hidden_layers']):
            base = PREFIX + f'layers.{layer}.decoder_layer.'
            if pending is None:
                n = self.norm(residual, base+'input_layernorm')
            else:
                residual, n = self.add_norm(residual, pending, base+'input_layernorm')
            attn = base+'self_attn.'
            q = self.qk(self.linear(n, attn+'q_proj'), attn+'q_norm', position, qh)
            k = self.qk(self.linear(n, attn+'k_proj'), attn+'k_norm', position, kh)
            v = self.linear(n, attn+'v_proj').reshape(rows, kh, dim)
            actual_k, actual_v = cache(layer)
            kv_errors.append([error(actual_k[position:], k), error(actual_v[position:], v)])
            keys = torch.cat((actual_k[:position], k)).repeat_interleave(qh//kh, 1).transpose(0, 1)
            values = torch.cat((actual_v[:position], v)).repeat_interleave(qh//kh, 1).transpose(0, 1)
            scores = q.transpose(0, 1).float() @ keys.transpose(1, 2).float() / dim**.5
            low = torch.arange(position, position+rows, device='cuda') - c['sliding_window'] + 1
            scores.masked_fill_(torch.arange(position+rows, device='cuda')[None, None, :] < low[None, :, None], -torch.inf)
            output = (scores.softmax(-1) @ values.float()).transpose(0, 1).reshape(rows, -1).to(torch.bfloat16)
            attention = self.fold(self.parts(output, attn+'o_proj'))
            residual, n = self.add_norm(residual, attention, base+'post_attention_layernorm')
            pending = self.fold(self.mlp(n, base+'mlp.'))
        _, hidden = self.add_norm(residual, pending, PREFIX+'final_norm')
        return hidden, kv_errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--engine-dir', type=Path, required=True)
    ap.add_argument('--sglang-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--world', type=int, default=4)
    ap.add_argument('--sampler-only', action='store_true')
    ap.add_argument('--blocks', type=int, default=0, help='0 checks every captured block')
    args = ap.parse_args()
    ref = DraftReference(args.checkpoint_dir/'dspark', args.world, args.sglang_source, native_cache=False)
    target = Weights(args.checkpoint_dir)
    def target_tensor(name):
        return torch.from_numpy(np.array(target.raw(name), copy=True)).view(torch.bfloat16).cuda()
    head = target_tensor('language_model.lm_head.weight')
    embed = None if args.sampler_only else target_tensor('language_model.model.embed_tokens.weight')
    pred = ref.tensor(PREFIX+'markov_head.markov_w1.weight')
    succ = ref.tensor(PREFIX+'markov_head.markov_w2.weight')
    c, results = ref.c, []
    rows, hidden = c['dflash_config']['block_size'], c['hidden_size']
    kh, dim = c['num_key_value_heads'], c['head_dim']
    def walk(logits, anchor, mode):
        previous, out = anchor, []
        for row in logits:
            bias = torch.mv(succ.float(), pred[previous].float())
            if mode == 'bf16':
                score = (row.to(torch.bfloat16).float() + bias.to(torch.bfloat16).float()).to(torch.bfloat16).float()
            else:
                score = (row.to(torch.bfloat16).float() if mode == 'bf16-base' else row.float()) + bias
            previous = int(score.argmax())
            out.append(previous)
        return out
    # Keep the 200K x 256 FP32 Markov matrix, avoiding seven conversions per block.
    succ = succ.float()
    for case in sorted(args.engine_dir.iterdir()):
        if not (case/'rank0').is_dir():
            continue
        roots = [case/f'rank{r}' for r in range(args.world)]
        blocks = sorted(roots[0].glob('block-*'), key=lambda p: int(p.name.split('-')[1]))
        for block in blocks[:args.blocks or None]:
            meta = json.loads((block/'metadata.json').read_text())
            actual = load(block/'hidden.bf16', np.uint16, (rows, hidden))
            logits = torch.cat([load(root/block.name/'logits.f32', np.float32, (rows, -1)) for root in roots], -1)
            proposals = np.fromfile(block/'proposals.i32', dtype=np.int32).tolist()
            targets = np.fromfile(block/'targets.i32', dtype=np.int32).tolist()
            projected = ref.mm(actual, head, rounded=False)
            record = dict(case=case.name, block=block.name, **meta, proposals=proposals, targets=targets,
                          head_relative_rms=error(logits, projected), predictions={})
            for mode in ('fp32', 'bf16-base', 'bf16'):
                record['predictions'][mode] = walk(projected, meta['anchor'], mode)
            if not args.sampler_only:
                def cache(layer):
                    return tuple(torch.cat([load(root/block.name/f'{name}-{layer}.bf16', np.uint16,
                               (meta['position']+rows, kh//args.world, dim)) for root in roots], 1) for name in ('k','v'))
                expected, kv_errors = ref.block(embed, meta, cache)
                record['hidden_relative_rms'] = [error(a, b) for a, b in zip(actual, expected)]
                record['block_kv_relative_rms'] = kv_errors
                expected_logits = ref.mm(expected, head, rounded=False)
                record['predictions']['reference-bf16-base'] = walk(expected_logits, meta['anchor'], 'bf16-base')
                record['predictions']['reference-bf16'] = walk(expected_logits, meta['anchor'], 'bf16')
            results.append(record)
            print(case.name, block.name, 'accepted', meta['accepted'], 'head', record['head_relative_rms'],
                  'hidden', record.get('hidden_relative_rms'), 'predictions', record['predictions'], flush=True)
            args.output.write_text(json.dumps(dict(upstream_full_stack=False,
                source='SGLang 58484246 native quantizers; independent PyTorch backbone and Markov arithmetic',
                cases=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
