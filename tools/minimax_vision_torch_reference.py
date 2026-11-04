#!/usr/bin/env python3
"""CUDA BF16 MiniMax vision reference; full-depth 0.5% RMS / .99998 cosine gate.

Explicit attention retains FP32 scores and BF16 probabilities, as in dgpp's
documented vision contract. --attention flash checks PyTorch FlashAttention
separately; it is not substituted silently for the explicit oracle.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from minimax_reference_dump import Weights, f32


def run(a):
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    c = json.loads((a.checkpoint_dir / 'config.json').read_text())['vision_config']
    source = Weights(a.checkpoint_dir)
    w = {}
    for name in source.header:
        if name.startswith(('vision_tower.', 'multi_modal_projector.', 'patch_merge_mlp.')):
            raw = torch.from_numpy(np.array(source.raw(name)))
            if source.header[name]['dtype'] == 'BF16': raw = raw.view(torch.bfloat16)
            w[name] = raw.cuda()
    H, heads = c['hidden_size'], c['num_attention_heads']
    D, width, height = H // heads, a.width, a.height
    n = width // 14 * (height // 14)
    p = 'vision_tower.vision_model.'
    def compare(name, x):
        path = a.engine_dir / (name + '.bf16')
        if not path.exists(): return
        actual = torch.from_numpy(f32(np.fromfile(path, dtype='<u2'))).cuda().reshape(x.shape)
        diff = actual - x.float()
        rel = (diff.square().mean().sqrt() / x.float().square().mean().sqrt()).item()
        cos = F.cosine_similarity(actual.flatten(), x.float().flatten(), dim=0).item()
        print(f'{name}: rel {rel:.8g}, cosine {cos:.9g}, changed {torch.count_nonzero(diff).item()}/{x.numel()}', flush=True)
        if a.trace_dir:
            a.trace_dir.mkdir(exist_ok=True, parents=True)
            (a.trace_dir / (name + '.bf16')).write_bytes(x.contiguous().cpu().view(torch.uint16).numpy().tobytes())
        return rel, cos
    y, x = np.indices((height, width))
    rgb = np.stack([x * 255 // (width - 1), y * 255 // (height - 1),
                    (x + y) * 255 // (width + height - 2)], -1).astype(np.uint8)
    if a.input_rgb:
        rgb = np.fromfile(a.input_rgb, dtype=np.uint8).reshape(height, width, 3)
    pixels = torch.from_numpy(rgb).cuda().float() / 255
    mean = torch.tensor([.48145466, .4578275, .40821073], device='cuda')
    std = torch.tensor([.26862954, .26130258, .27577711], device='cuda')
    pixels = (pixels - mean) / std
    frames = torch.stack([pixels, pixels]).permute(0, 3, 1, 2)
    patches = frames.reshape(1, 2, 3, height // 28, 2, 14, width // 28, 2, 14)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(n, -1).bfloat16()
    x = F.linear(patches.float(), w[p + 'embeddings.patch_embedding.weight'].reshape(H, -1)).bfloat16()
    compare('patch_projection', x)
    def ln(x, name): return F.layer_norm(x, (H,), w[name + '.weight'], w[name + '.bias'], c['layer_norm_eps'])
    def linear(x, name): return F.linear(x, w[name + '.weight'], w[name + '.bias'])
    x = ln(x, p + 'pre_layrnorm')
    compare('pre_norm', x)
    axis = D // 6
    pos = [[0, by * 2 + iy, bx * 2 + ix] for by in range(height // 28) for bx in range(width // 28)
           for iy in range(2) for ix in range(2)]
    freq = 1 / torch.pow(c['rope_theta'], torch.arange(axis, device='cuda', dtype=torch.float32) / axis)
    angles = (torch.tensor(pos, device='cuda')[:, :, None] * freq).reshape(n, 3 * axis)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    rope_path = a.engine_dir / 'rope.f32'
    if rope_path.exists():
        actual = np.fromfile(rope_path, dtype='<f4')
        expected = torch.cat([cos.flatten(), sin.flatten()]).cpu().numpy()
        print(f'rope table: max error {np.max(np.abs(actual - expected))}, changed {np.count_nonzero(actual != expected)}/{actual.size}', flush=True)
        if not np.array_equal(actual, expected):
            raise AssertionError('FP32 rotary tables differ from the CUDA reference')
    def rotate(q):
        q = q.float()
        left, right = q[..., :3 * axis], q[..., 3 * axis:6 * axis]
        return torch.cat([left * cos - right * sin, right * cos + left * sin, q[..., 6 * axis:]], -1).bfloat16()
    for layer in range(c['num_hidden_layers']):
        base = p + f'encoder.layers.{layer}.'
        norm = ln(x, base + 'layer_norm1')
        qkvw = torch.cat([w[base + f'self_attn.{z}_proj.weight'] for z in 'qkv'])
        qkvb = torch.cat([w[base + f'self_attn.{z}_proj.bias'] for z in 'qkv'])
        qkv = F.linear(norm, qkvw, qkvb)
        compare(f'layer{layer}.qkv', qkv)
        q, k, v = qkv.reshape(n, 3, heads, D).unbind(1)
        q, k = rotate(q), rotate(k)
        q, k, v = [z.transpose(0, 1).contiguous() for z in (q, k, v)]
        compare(f'layer{layer}.q', q)
        compare(f'layer{layer}.k', k)
        if a.attention == 'flash':
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
        else:
            scores = torch.bmm(q, k.transpose(1, 2), out_dtype=torch.float32)
            probabilities = (scores * (D ** -.5)).softmax(-1).bfloat16()
            out = torch.bmm(probabilities, v)
        out = out.transpose(0, 1).reshape(n, H)
        compare(f'layer{layer}.attn', out)
        projected = linear(out, base + 'self_attn.out_proj')
        compare(f'layer{layer}.attn_proj', projected)
        x = x + projected
        compare(f'layer{layer}.attn_residual', x)
        norm = ln(x, base + 'layer_norm2')
        compare(f'layer{layer}.norm2', norm)
        mlp = linear(norm, base + 'mlp.fc1')
        compare(f'layer{layer}.mlp_hidden', mlp)
        mlp = F.gelu(mlp)
        compare(f'layer{layer}.mlp_act', mlp)
        mlp = linear(mlp, base + 'mlp.fc2')
        compare(f'layer{layer}.mlp_proj', mlp)
        x = x + mlp
        compare(f'layer{layer}', x)
    x = linear(F.gelu(linear(x, 'multi_modal_projector.linear_1')), 'multi_modal_projector.linear_2')
    compare('projected_patches', x)
    x = linear(F.gelu(linear(x.reshape(n // 4, -1), 'patch_merge_mlp.linear_1')), 'patch_merge_mlp.linear_2')
    rel, cosine = compare('image_rows', x)
    if not np.isfinite(rel) or rel > .005 or cosine < .99998:
        raise AssertionError('full-depth vision numerical gate failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--engine-dir', type=Path, required=True)
    ap.add_argument('--width', type=int, default=56)
    ap.add_argument('--height', type=int, default=56)
    ap.add_argument('--trace-dir', type=Path)
    ap.add_argument('--input-rgb', type=Path, help='Processed RGB pixels used by the native harness')
    ap.add_argument('--attention', choices=['explicit', 'flash'], default='explicit')
    run(ap.parse_args())
