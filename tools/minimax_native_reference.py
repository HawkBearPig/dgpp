#!/usr/bin/env python3
"""Independent MiniMax native-precision reference, using pinned GPU quantizers.

This is a measurement instrument, not an upstream serving stack. SGLang's
MXFP8 producer and FlashInfer's dynamic NVFP4 producer are executed directly;
matrix products use independently decoded operands and FP32 accumulation.
MSA attention uses its independently cross-checked precision formula. Both
upstream attention phase formulas can be selected explicitly. No DGPP CUDA
kernel or GEMM is imported. See docs/minimax_native_precision.md.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

import minimax_native_attention_reference as msa
from minimax_mxfp8_quantization_check import SOURCE as MX_SOURCE, SHA256 as MX_SHA256
from minimax_torch_reference import Reference


class NativeReference(Reference):
    def __init__(self, checkpoint, world, sglang_source, phase='prefill',
                 native_activations=True, native_cache=True, projection_backend='cublas',
                 elementwise_backend='torch'):
        super().__init__(checkpoint, world, False, projection_backend=projection_backend,
                         elementwise_backend=elementwise_backend,
                         attention_backend=f'native-{phase}' if native_cache else 'torch',
                         activation_quantization='native' if native_activations else 'bf16')
        self.native_activations = native_activations
        self.native_cache = native_cache
        self.phase = phase
        if native_activations:
            path = sglang_source / MX_SOURCE
            if hashlib.sha256(path.read_bytes()).hexdigest() != MX_SHA256:
                raise ValueError('expected the pinned SGLang 58484246 MXFP8 producer')
            spec = importlib.util.spec_from_file_location('_minimax_native_reference_mx', path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.mx_quant = module.mxfp8_quantize_swizzled
            # Check the actual producer files, then keep all tensors on GPU.
            from minimax_native_moe_regression import upstream_producer
            upstream_producer()
            from flashinfer.quantization.kernels.nvfp4_quantize import nvfp4_quantize_per_token_cute_dsl
            self.nv_quant = nvfp4_quantize_per_token_cute_dsl
            self.nv_global_inv = torch.tensor([1 / (6 * 448)], device='cuda', dtype=torch.float32)

    def quantized(self, x, mode):
        if x.dtype != torch.bfloat16:
            raise ValueError('native projection input must have the specified BF16 boundary')
        x = x.contiguous()
        if mode == 'mxfp8':
            codes, scales = self.mx_quant(x)
            r = torch.arange(len(x), device=x.device)[:, None]
            c = torch.arange(x.shape[1] // 32, device=x.device)[None, :]
            blocks = (x.shape[1] // 32 + 3) // 4
            offsets = ((r // 128) * blocks + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
            scale = torch.exp2(scales[offsets].float() - 127).repeat_interleave(32, -1)
            return (codes.float() * scale).to(torch.bfloat16), None
        codes, scales, globals_ = self.nv_quant(
            x, self.nv_global_inv, sf_layout=2, enable_pdl=False, nvfp4_4over6=None)
        codes = torch.stack((codes & 15, codes >> 4), -1).flatten(-2)
        decoded = self.fp4[codes.long()] * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        return decoded.to(torch.bfloat16), globals_[:, None]

    def project(self, x, weight, divisor, base, rounded=True):
        if not self.native_activations:
            return super().project(x, weight, divisor, base, rounded)
        mode = ('mxfp8' if base + '.weight_scale_inv' in self.w.header else
                'nvfp4' if base + '.weight_packed' in self.w.header else None)
        if mode is None:
            return self.mm(x, weight, divisor, rounded)
        x, scale = self.quantized(x, mode)
        y = self.mm(x, weight, rounded=False, streaming_order=mode == 'mxfp8')
        if scale is not None:
            # Per-token decode scale and reciprocal weight encode scale are
            # one FP32 epilogue factor, outside the block-scaled dot.
            inverse = torch.tensor(divisor, device=x.device, dtype=torch.float32).reciprocal()
            y = y * (scale * inverse)
        return y.to(torch.bfloat16) if rounded else y

    def mlp(self, n, base, expert=False):
        if not self.native_activations:
            return super().mlp(n, base, expert)
        c = self.c
        gate, up, down = ('w1', 'w3', 'w2') if expert else ('gate_proj', 'up_proj', 'down_proj')
        g = self.project(n, *self.matrix(base + gate), base + gate, rounded=not expert)
        u = self.project(n, *self.matrix(base + up), base + up, rounded=not expert)
        if self.elementwise_backend == 'triton-fixed':
            from minimax_reference_arithmetic import activation
            act = activation(g, u, c['swiglu_alpha'], c['swiglu_limit'], rounded=True)
        else:
            g = g.float().clamp(max=c['swiglu_limit'])
            u = u.float().clamp(-c['swiglu_limit'], c['swiglu_limit'])
            act = (g / (1 + torch.exp(-c['swiglu_alpha'] * g)) * (u + 1)).to(torch.bfloat16)
        return self.parts(act, base + down, rounded=True)

    def attention(self, n, layer):
        if not self.native_cache:
            return super().attention(n, layer)
        c, rows = self.c, len(n)
        qh, kh, dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        ih = c['sparse_attention_config']['sparse_num_index_heads']
        topk = c['sparse_attention_config']['sparse_topk_blocks']
        base = f'language_model.model.layers.{layer}.self_attn.'
        def project(name, heads, normalize=True):
            value = self.linear(n, base + name + '_proj').reshape(rows, heads, dim)
            return self.rotary(self.norm(value, base + name + '_norm')) if normalize else value
        q, k, v = project('q', qh), project('k', kh), project('v', kh, False)
        iq, ik = project('index_q', ih), project('index_k', 1)
        q, iq = msa.queries(q), msa.queries(iq)
        k, v, ik = (msa.unpack_cache(*msa.pack_cache(value)) for value in (k, v, ik))
        positions = torch.arange(rows, device=n.device, dtype=torch.int64)
        selected = msa.select_blocks(iq, ik, positions, topk)
        output = msa.attention(q, k, v, selected, positions, self.phase)
        return self.fold(self.parts(output.flatten(1), base + 'o_proj'))

    def check_stages(self, engine, output, layers=0):
        metadata = json.loads((engine / 'metadata.json').read_text())
        if metadata.get('activations', 'bf16') != ('native' if self.native_activations else 'bf16'):
            raise ValueError('engine/reference activation recipes differ')
        if metadata.get('kv_dtype', 'bf16') != ('fp4' if self.native_cache else 'bf16'):
            raise ValueError('engine/reference cache recipes differ')
        return super().check_stages(engine, output, layers)

    def run(self, cases, output, dump_states=False):
        super().run(cases, output, dump_states)
        for case in cases:
            path = output / case['name'] / 'metadata.json'
            metadata = json.loads(path.read_text())
            metadata.update(
                backend='independent decoded-operand products with pinned native GPU quantizers',
                native_activations=self.native_activations,
                native_cache=self.native_cache,
                attention_phase=self.phase if self.native_cache else None,
                sglang_revision='58484246d26523eaf1a7f16ab524deee288d2120',
                flashinfer_revision='d84335d8c54e0c8f9f7e22faa15c7e34a58abf94',
                msa_revision='c8e1013a16a708d2a7656805d5fc26f2b87715d1',
                upstream_full_stack=False)
            path.write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--sglang-source', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', type=Path)
    mode.add_argument('--check-engine-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--attention-phase', choices=('prefill', 'decode'), default='prefill')
    parser.add_argument('--bf16-activations', action='store_true')
    parser.add_argument('--bf16-cache', action='store_true')
    parser.add_argument('--projection-backend', choices=('cublas', 'triton-fixed'), default='cublas')
    parser.add_argument('--elementwise-backend', choices=('torch', 'triton-fixed'), default='torch')
    parser.add_argument('--dump-states', action='store_true')
    parser.add_argument('--check-layers', type=int, default=0)
    args = parser.parse_args()
    reference = NativeReference(args.checkpoint_dir, args.world, args.sglang_source,
                                args.attention_phase, not args.bf16_activations, not args.bf16_cache,
                                args.projection_backend, args.elementwise_backend)
    with torch.inference_mode():
        if args.check_engine_dir:
            reference.check_stages(args.check_engine_dir, args.output, args.check_layers)
            return
        if args.check_layers:
            parser.error('--check-layers requires --check-engine-dir')
        cases = json.loads(args.manifest.read_text())
        tokenizer = None
        for case in cases:
            if 'ids' not in case:
                if tokenizer is None:
                    tokenizer = Tokenizer.from_file(str(args.checkpoint_dir / 'tokenizer.json'))
                case['ids'] = tokenizer.encode(case['text'], add_special_tokens=False).ids
            if len(case['ids']) < 2 or Path(case['name']).name != case['name']:
                parser.error('each case needs a simple name and at least two tokens')
            score_from = case.get('score_from', 1)
            if type(score_from) is not int or not 1 <= score_from < len(case['ids']):
                parser.error('score_from must select a target token after the first input token')
        reference.run(cases, args.output, args.dump_states)


if __name__ == '__main__':
    main()
