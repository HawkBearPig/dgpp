#!/usr/bin/env python3
"""Independent FP64 MiniMax vision oracle, with explicit BF16 boundaries.

The tensor inventory and equations follow the pinned MiniMax/vLLM processor
and tower; no dgpp binding or kernel helper defines the visual operations.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from minimax_reference_dump import Weights, f32, rounded, relative_rms


def check(path, dump, width, height, isolate_layers=False, report_all=False, frames_count=1):
    c = json.loads((path / 'config.json').read_text())['vision_config']
    weights = Weights(path)
    h, heads, depth = c['hidden_size'], c['num_attention_heads'], c['num_hidden_layers']
    temporal = max(1, frames_count // 2)
    d, n = h // heads, width // 14 * (height // 14) * temporal
    failures = []
    def read(name, shape):
        return f32(np.fromfile(dump / (name + '.bf16'), dtype='<u2')).reshape(shape)
    def compare(name, expected, limit=.008, dtype='<u2'):
        raw = np.fromfile(dump / (name + ('.f32' if dtype == '<f4' else '.bf16')), dtype=dtype)
        actual = (raw if dtype == '<f4' else f32(raw)).reshape(expected.shape)
        error = relative_rms(actual, expected)
        print(f'{name}: relative RMS {error:.7g}', flush=True)
        if not np.isfinite(error) or error > limit:
            failures.append(f'{name}: {error} > {limit}')
            if not report_all:
                raise AssertionError(failures[-1])
        return actual
    y, x = np.indices((height, width))
    rgb = np.stack([x * 255 // (width - 1), y * 255 // (height - 1),
                    (x + y) * 255 // (width + height - 2)], axis=-1).astype(np.uint8)
    rgb = np.stack([(rgb.astype(np.uint16) + np.array([17, 29, 43]) * i).astype(np.uint8)
                    for i in range(frames_count)])
    mean = np.array([.48145466, .4578275, .40821073], dtype=np.float32)
    std = np.array([.26862954, .26130258, .27577711], dtype=np.float32)
    pixels = (rgb.astype(np.float32) * np.float32(1 / 255) - mean) / std
    # NCHW, duplicate temporal frame; block, spatial merge, then C/T/P/P.
    frames = (np.repeat(pixels, 2, axis=0) if frames_count == 1 else pixels).transpose(0, 3, 1, 2)
    patches = frames.reshape(temporal, 2, 3, height // 28, 2, 14, width // 28, 2, 14)
    patches = rounded(patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(n, -1))
    compare('patches', patches, 0, '<f4')
    p = 'vision_tower.vision_model.'
    conv = weights.tensor(p + 'embeddings.patch_embedding.weight').reshape(h, -1)
    x = rounded(patches.astype(np.float64) @ conv.astype(np.float64).T)
    compare('patch_projection', x, .0003)
    def ln(x, name):
        a = x.astype(np.float64)
        normalized = (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + c['layer_norm_eps'])
        return rounded(normalized * weights.tensor(name + '.weight') + weights.tensor(name + '.bias'))
    def linear(x, name):
        return rounded(x.astype(np.float64) @ weights.tensor(name + '.weight').astype(np.float64).T +
                       weights.tensor(name + '.bias').astype(np.float64))
    def gelu(x):
        a = x.astype(np.float64)
        return rounded(.5 * a * (1 + np.vectorize(math.erf)(a / np.sqrt(2))))
    x = ln(x, p + 'pre_layrnorm')
    compare('pre_norm', x, .001)
    axis = d // 6
    positions = [[t, by * 2 + iy, bx * 2 + ix] for t in range(temporal)
                 for by in range(height // 28) for bx in range(width // 28)
                 for iy in range(2) for ix in range(2)]
    angles = (np.asarray(positions, dtype=np.float32)[:, :, None] *
              np.power(c['rope_theta'], -np.arange(axis, dtype=np.float32) / axis)).reshape(n, 3 * axis)
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    def rotate(q):
        q = q.reshape(n, heads, d)
        a, b = q[..., :3 * axis], q[..., 3 * axis:6 * axis]
        return rounded(np.concatenate([a * cos - b * sin, b * cos + a * sin,
                                       q[..., 6 * axis:]], -1)).transpose(1, 0, 2)
    for l in range(depth):
        if isolate_layers:
            x = read('pre_norm' if l == 0 else f'layer{l-1}', (n, h))
        base = p + f'encoder.layers.{l}.'
        norm = ln(x, base + 'layer_norm1')
        compare(f'layer{l}.norm1', norm)
        projections = [linear(norm, base + 'self_attn.' + proj + '_proj') for proj in ('q', 'k', 'v')]
        compare(f'layer{l}.qkv', np.concatenate(projections, -1))
        q, k = map(rotate, projections[:2])
        v = projections[2].reshape(n, heads, d).transpose(1, 0, 2)
        for name, value in [('q', q), ('k', k), ('v', v)]: compare(f'layer{l}.{name}', value)
        scores = (q.astype(np.float64) @ k.astype(np.float64).transpose(0, 2, 1)) / np.sqrt(d)
        probabilities = np.exp(scores - scores.max(-1, keepdims=True))
        probabilities /= probabilities.sum(-1, keepdims=True)
        attention = rounded(rounded(probabilities).astype(np.float64) @ v.astype(np.float64))
        attention = attention.transpose(1, 0, 2).reshape(n, h)
        compare(f'layer{l}.attn', attention)
        x = rounded(x + linear(attention, base + 'self_attn.out_proj'))
        compare(f'layer{l}.attn_residual', x)
        mlp = linear(gelu(linear(ln(x, base + 'layer_norm2'), base + 'mlp.fc1')), base + 'mlp.fc2')
        x = rounded(x + mlp)
        compare(f'layer{l}', x)
    if isolate_layers:
        x = read(f'layer{depth-1}', (n, h))
    projected = linear(gelu(linear(x, 'multi_modal_projector.linear_1')), 'multi_modal_projector.linear_2')
    compare('projected_patches', projected)
    if isolate_layers:
        projected = read('projected_patches', projected.shape)
    out = linear(gelu(linear(projected.reshape(n // 4, -1), 'patch_merge_mlp.linear_1')), 'patch_merge_mlp.linear_2')
    compare('image_rows', out)
    if failures:
        raise AssertionError(f'{len(failures)} failed boundaries; first: {failures[0]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--engine-dir', type=Path, required=True)
    ap.add_argument('--width', type=int, default=84)
    ap.add_argument('--height', type=int, default=56)
    ap.add_argument('--frames', type=int, choices=[1, 2, 4, 6, 8], default=1)
    ap.add_argument('--isolate-layers', action='store_true', help='Check each layer on observed inputs; not an end-to-end quality gate')
    ap.add_argument('--report-all', action='store_true', help='Report every boundary before returning a failing exit status')
    a = ap.parse_args()
    check(a.checkpoint_dir, a.engine_dir, a.width, a.height, a.isolate_layers, a.report_all, a.frames)
