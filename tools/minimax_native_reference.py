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
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import minimax_native_attention_reference as msa
from minimax_mxfp8_quantization_check import load_producers
from minimax_torch_reference import Reference


class NativeReference(Reference):
    def __init__(self, checkpoint, world, sglang_source, phase='prefill',
                 native_activations=True, native_cache=True, projection_backend='cublas',
                 elementwise_backend='torch', attention_arithmetic='torch'):
        super().__init__(checkpoint, world, False, projection_backend=projection_backend,
                         elementwise_backend=elementwise_backend,
                         attention_backend=f'native-{phase}-{attention_arithmetic}' if native_cache else 'torch',
                         activation_quantization='native' if native_activations else 'bf16')
        self.native_activations = native_activations
        self.native_cache = native_cache
        self.phase = phase
        self.attention_arithmetic = attention_arithmetic
        if attention_arithmetic == 'triton-fixed' and not native_cache:
            raise ValueError('the fixed native attention control requires native cache')
        if native_activations:
            module, fused = load_producers(sglang_source)
            self.mx_quant = module.mxfp8_quantize_swizzled
            self.mx_swiglu = fused.swiglu_oai_mxfp8
            # Check the actual producer files, then keep all tensors on GPU.
            from minimax_native_moe_regression import upstream_producer
            upstream_producer()
            from flashinfer.quantization.kernels.nvfp4_quantize import nvfp4_quantize_per_token_cute_dsl
            self.nv_quant = nvfp4_quantize_per_token_cute_dsl
            self.nv_global_inv = torch.tensor([1 / (6 * 448)], device='cuda', dtype=torch.float32)

    @staticmethod
    def mx_scales(codes, scales):
        r = torch.arange(len(codes), device=codes.device)[:, None]
        c = torch.arange(codes.shape[1] // 32, device=codes.device)[None, :]
        blocks = (codes.shape[1] // 32 + 3) // 4
        offsets = ((r // 128) * blocks + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
        return scales[offsets].contiguous()

    @classmethod
    def decode_mx(cls, codes, scales):
        scale = torch.exp2(cls.mx_scales(codes, scales).float() - 127).repeat_interleave(32, -1)
        return (codes.float() * scale).to(torch.bfloat16)

    def mm(self, x, weight, divisor=1.0, rounded=True, streaming_order=False):
        if self.projection_backend != 'triton-native':
            return super().mm(x, weight, divisor, rounded, streaming_order)
        # Unquantized matrices (notably the vocabulary head) use the existing
        # independently checked BF16 fixed-order product in this control.
        from minimax_reference_gemm import projection
        value = projection(x, weight, streaming_order) / divisor
        return value.to(torch.bfloat16) if rounded else value

    def native_product(self, codes, scales, globals_, base, mode, column, divisor, rounded):
        from minimax_native_reference_gemm import product
        packed = mode == 'nvfp4'
        inner = codes.shape[1] * (2 if packed else 1)
        name = base + ('.weight_packed' if packed else '.weight')
        start, width = (column // 2, inner // 2) if packed else (column, inner)
        raw = self.w.raw(name)[:, start:start+width]
        weight = torch.from_numpy(np.array(raw, copy=True)).cuda()
        name = base + ('.weight_scale' if packed else '.weight_scale_inv')
        group = 16 if packed else 32
        raw = self.w.raw(name)[:, column // group:(column+inner) // group]
        weight_scales = torch.from_numpy(np.array(raw, copy=True)).cuda()
        value = product(codes.contiguous().view(torch.uint8), scales.contiguous().view(torch.uint8),
                        weight, weight_scales, mode)
        if globals_ is not None:
            inverse = torch.tensor(divisor, device=value.device, dtype=torch.float32).reciprocal()
            value = value * (globals_[:, None] * inverse)
        return value.to(torch.bfloat16) if rounded else value

    def quantized(self, x, mode):
        if x.dtype != torch.bfloat16:
            raise ValueError('native projection input must have the specified BF16 boundary')
        x = x.contiguous()
        if mode == 'mxfp8':
            return self.decode_mx(*self.mx_quant(x)), None
        codes, scales, globals_ = self.nv_quant(
            x, self.nv_global_inv, sf_layout=2, enable_pdl=False, nvfp4_4over6=None)
        codes = torch.stack((codes & 15, codes >> 4), -1).flatten(-2)
        decoded = self.fp4[codes.long()] * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        return decoded.to(torch.bfloat16), globals_[:, None]

    def project(self, x, weight, divisor, base, rounded=True, column=0):
        if not self.native_activations:
            return super().project(x, weight, divisor, base, rounded)
        mode = ('mxfp8' if base + '.weight_scale_inv' in self.w.header else
                'nvfp4' if base + '.weight_packed' in self.w.header else None)
        if mode is None:
            return self.mm(x, weight, divisor, rounded)
        if self.projection_backend == 'triton-native':
            x = x.contiguous()
            if mode == 'mxfp8':
                codes, scales = self.mx_quant(x)
                scales, globals_ = self.mx_scales(codes, scales), None
            else:
                codes, scales, globals_ = self.nv_quant(
                    x, self.nv_global_inv, sf_layout=2, enable_pdl=False, nvfp4_4over6=None)
            return self.native_product(codes, scales, globals_, base, mode, column, divisor, rounded)
        x, scale = self.quantized(x, mode)
        # The native GEMM walks K in ordinary order. The BF16 control's
        # streaming-weight column permutation does not apply to this kernel.
        y = self.mm(x, weight, rounded=False)
        if scale is not None:
            # Per-token decode scale and reciprocal weight encode scale are
            # one FP32 epilogue factor, outside the block-scaled dot.
            inverse = torch.tensor(divisor, device=x.device, dtype=torch.float32).reciprocal()
            y = y * (scale * inverse)
        return y.to(torch.bfloat16) if rounded else y

    def parts(self, x, base, rounded=True):
        if not self.native_activations or self.projection_backend != 'triton-native':
            return super().parts(x, base, rounded)
        weight, divisor = self.matrix(base)
        width = weight.shape[1] // self.world
        return [self.project(x[:, r*width:(r+1)*width].contiguous(),
                             weight[:, r*width:(r+1)*width].contiguous(), divisor, base,
                             rounded, column=r*width) for r in range(self.world)]

    def mlp(self, n, base, expert=False):
        if not self.native_activations:
            return super().mlp(n, base, expert)
        c = self.c
        gate, up, down = ('w1', 'w3', 'w2') if expert else ('gate_proj', 'up_proj', 'down_proj')
        g = self.project(n, *self.matrix(base + gate), base + gate, rounded=not expert)
        u = self.project(n, *self.matrix(base + up), base + up, rounded=not expert)
        if not expert:
            # Execute the actual upstream fusion, including its sigmoid
            # evaluation, BF16 boundary and MXFP8 rounding. Group scales are
            # local to 32 values, so slicing after quantization preserves TP.
            codes, scales = self.mx_swiglu(torch.cat((g, u), dim=1),
                                           c['swiglu_alpha'], c['swiglu_limit'])
            weight, divisor = self.matrix(base + down)
            if divisor != 1.0:
                raise ValueError('dense/shared down projection must be MXFP8')
            width = weight.shape[1] // self.world
            if self.projection_backend == 'triton-native':
                scales = self.mx_scales(codes, scales)
                return [self.native_product(codes[:, r*width:(r+1)*width],
                                             scales[:, r*width//32:(r+1)*width//32], None,
                                             base+down, 'mxfp8', r*width, divisor, True)
                        for r in range(self.world)]
            act = self.decode_mx(codes, scales)
            return [self.mm(act[:, r*width:(r+1)*width].contiguous(),
                            weight[:, r*width:(r+1)*width].contiguous())
                    for r in range(self.world)]
        if self.elementwise_backend == 'triton-fixed':
            from minimax_reference_arithmetic import activation
            act = activation(g, u, c['swiglu_alpha'], c['swiglu_limit'], rounded=True)
        else:
            g = g.float().clamp(max=c['swiglu_limit'])
            u = u.float().clamp(-c['swiglu_limit'], c['swiglu_limit'])
            act = (g / (1 + torch.exp(-c['swiglu_alpha'] * g)) * (u + 1)).to(torch.bfloat16)
        return self.parts(act, base + down, rounded=True)

    def attention(self, n, layer, *, prefill_tokens=None):
        if not self.native_cache:
            return super().attention(n, layer, prefill_tokens=prefill_tokens)
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
        if self.attention_arithmetic == 'triton-fixed':
            import minimax_native_reference_attention as fixed
            selected = msa.select_blocks(iq, ik, positions, topk, score_fn=fixed.scores)
            calculate = fixed.attention
        else:
            selected = msa.select_blocks(iq, ik, positions, topk)
            calculate = msa.attention
        if self.phase == 'upstream':
            if prefill_tokens is None or not 1 <= prefill_tokens <= rows:
                raise ValueError('upstream phase requires the manifest prompt boundary')
            # A prompt of P tokens produces the first scored target from row
            # P-1 using prefill. Rows P onward consume continuation tokens.
            output = torch.cat([
                calculate(q[:prefill_tokens], k, v, selected[:prefill_tokens],
                          positions[:prefill_tokens], phase='prefill'),
                calculate(q[prefill_tokens:], k, v, selected[prefill_tokens:],
                          positions[prefill_tokens:], phase='decode')
            ]) if prefill_tokens < rows else calculate(q, k, v, selected, positions, phase='prefill')
        else:
            output = calculate(q, k, v, selected, positions, phase=self.phase)
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
                backend=('independent Triton native-product order control with pinned GPU quantizers'
                         if self.projection_backend == 'triton-native' else
                         'independent decoded-operand products with pinned native GPU quantizers'),
                native_activations=self.native_activations,
                native_cache=self.native_cache,
                attention_phase=self.phase if self.native_cache else None,
                attention_arithmetic=self.attention_arithmetic,
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
    parser.add_argument('--attention-phase', choices=('prefill', 'decode', 'upstream'), default='prefill',
                        help='upstream uses prefill for the manifest prompt and decode for its continuation')
    parser.add_argument('--attention-arithmetic', choices=('torch', 'triton-fixed'), default='torch')
    parser.add_argument('--bf16-activations', action='store_true')
    parser.add_argument('--bf16-cache', action='store_true')
    parser.add_argument('--projection-backend', choices=('cublas', 'triton-fixed', 'triton-native'), default='cublas')
    parser.add_argument('--elementwise-backend', choices=('torch', 'triton-fixed'), default='torch')
    parser.add_argument('--dump-states', action='store_true')
    parser.add_argument('--check-layers', type=int, default=0)
    args = parser.parse_args()
    if args.check_engine_dir and args.attention_phase == 'upstream':
        parser.error('--attention-phase upstream requires --manifest with a prompt boundary')
    reference = NativeReference(args.checkpoint_dir, args.world, args.sglang_source,
                                args.attention_phase, not args.bf16_activations, not args.bf16_cache,
                                args.projection_backend, args.elementwise_backend, args.attention_arithmetic)
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
