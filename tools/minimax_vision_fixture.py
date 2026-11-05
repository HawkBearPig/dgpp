#!/usr/bin/env python3
"""Independent MiniMax visual tensor inventory; FP32 patch weights remain FP32."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import struct


def write(path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'tests/data/minimax/config.json').read_text())
    config['vision_config'].update(hidden_size=160, num_attention_heads=2,
                                  num_hidden_layers=2, intermediate_size=256, projection_dim=256)
    config['text_config'].update(hidden_size=256, vocab_size=512)
    config.update(projector_hidden_size=256, image_token_index=500, video_token_index=501,
                  vision_start_token_id=502, vision_end_token_id=503)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    tensors = []
    def add(name, shape, dtype='BF16', kind='matrix'):
        tensors.append((name, shape, dtype, kind))
    def linear(name, n, k):
        add(name + '.weight', [n, k])
        add(name + '.bias', [n], kind='bias')
    def norm(name):
        add(name + '.weight', [160], kind='norm')
        add(name + '.bias', [160], kind='bias')
    p = 'vision_tower.vision_model.'
    add(p + 'embeddings.patch_embedding.weight', [160, 3, 2, 14, 14], 'F32')
    norm(p + 'pre_layrnorm')
    for l in range(2):
        base = p + f'encoder.layers.{l}.'
        norm(base + 'layer_norm1'); norm(base + 'layer_norm2')
        for proj in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            linear(base + 'self_attn.' + proj, 160, 160)
        linear(base + 'mlp.fc1', 256, 160)
        linear(base + 'mlp.fc2', 160, 256)
    linear('multi_modal_projector.linear_1', 256, 160)
    linear('multi_modal_projector.linear_2', 256, 256)
    linear('patch_merge_mlp.linear_1', 256, 1024)
    linear('patch_merge_mlp.linear_2', 256, 256)
    header, payload = {}, bytearray()
    for name, shape, dtype, kind in tensors:
        rng = random.Random(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little'))
        data = bytearray()
        scale = 0.5 / math.sqrt(math.prod(shape[1:])) if len(shape) > 1 else .04
        for _ in range(math.prod(shape)):
            value = rng.uniform(-scale, scale) + (1 if kind == 'norm' else 0)
            if dtype == 'F32':
                data.extend(struct.pack('<f', value))
            else:
                bits = struct.unpack('<I', struct.pack('<f', value))[0]
                data.extend(struct.pack('<H', (bits + 0x7fff + ((bits >> 16) & 1)) >> 16))
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload) + len(data)])
        payload.extend(data)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    (path / 'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    print(f'{len(tensors)} vision tensors, {len(payload)} bytes')


def combine(path, target):
    """Join independently generated text/draft and vision fixtures losslessly."""
    vision = json.loads((path / 'config.json').read_text())
    config = json.loads((target / 'config.json').read_text())
    for key in ('vision_config', 'img_token_compression_config', 'projector_hidden_size',
                'image_token_index', 'video_token_index', 'vision_start_token_id',
                'vision_end_token_id', 'multimodal_projector_bias', 'projector_hidden_act',
                'vision_feature_layer', 'vision_feature_select_strategy', 'process_image_mode'):
        config[key] = vision[key]
    assert config['text_config']['hidden_size'] == vision['text_config']['hidden_size']
    (path / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (path / 'model.safetensors').replace(path / 'vision.safetensors')
    shutil.copyfile(target / 'model.safetensors', path / 'text.safetensors')
    if (target / 'dspark').exists():
        shutil.copytree(target / 'dspark', path / 'dspark', dirs_exist_ok=True)
    weight_map = {}
    for file in ('vision.safetensors', 'text.safetensors'):
        with (path / file).open('rb') as f:
            length, = struct.unpack('<Q', f.read(8))
            header = json.loads(f.read(length))
        for key in header:
            if key != '__metadata__':
                assert key not in weight_map
                weight_map[key] = file
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--target-checkpoint', type=Path)
    a = ap.parse_args()
    write(a.output)
    if a.target_checkpoint:
        combine(a.output, a.target_checkpoint)
