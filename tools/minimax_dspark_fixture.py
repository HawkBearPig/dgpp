#!/usr/bin/env python3
"""Write a small DSpark checkpoint from an independent upstream tensor inventory."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import struct


def write(directory: Path, config_path: Path):
    c = json.loads(config_path.read_text())
    t = c['text_config']
    t.update(hidden_size=256, vocab_size=512, num_hidden_layers=2,
             num_attention_heads=64, num_key_value_heads=4,
             dense_intermediate_size=256, dspark_ffn_hidden_size=256,
             max_position_embeddings=4096, sliding_window=256,
             dspark_target_layer_ids=[1, 3], moe_layer_freq=[0, 0],
             layer_types=['sliding_attention'] * 2)
    t['dspark_config']['ffn_hidden_size'] = 256
    t['dflash_config'].update(num_hidden_layers=2, num_target_layers=4,
                             target_layer_ids=[1, 3], sliding_window=256,
                             layer_types=['sliding_attention'] * 2, mask_token_id=511)
    t['dspark_noise_token_id'] = 511
    t['dspark_config']['mask_token_id'] = 511
    for key in list(c):
        if key.startswith('dspark_') or key == 'dflash_config':
            c[key] = copy.deepcopy(t[key])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(c, indent=2) + '\n')
    tensors = []
    def add(name, dtype, shape):
        tensors.append(('language_model.model.dspark.' + name, dtype, shape))
    def matrix(name, rows, cols):
        add(name + '.weight', 'F8_E4M3', [rows, cols])
        add(name + '.weight_scale_inv', 'U8', [rows, cols // 32])
    for layer in range(2):
        p = f'layers.{layer}.decoder_layer.'
        for norm in ('input_layernorm', 'post_attention_layernorm', 'self_attn.q_norm', 'self_attn.k_norm'):
            add(p + norm + '.weight', 'BF16', [256] if 'self_attn' not in norm else [128])
        matrix(p + 'self_attn.q_proj', 8192, 256)
        matrix(p + 'self_attn.k_proj', 512, 256)
        matrix(p + 'self_attn.v_proj', 512, 256)
        matrix(p + 'self_attn.o_proj', 256, 8192)
        matrix(p + 'mlp.gate_proj', 256, 256)
        matrix(p + 'mlp.up_proj', 256, 256)
        matrix(p + 'mlp.down_proj', 256, 256)
    add('fc.weight', 'BF16', [256, 512])
    for norm in ('hidden_norm', 'final_norm'):
        add(norm + '.weight', 'BF16', [256])
    add('markov_head.markov_w1.weight', 'BF16', [512, 256])
    add('markov_head.markov_w2.weight', 'BF16', [512, 256])
    add('confidence_head.proj.weight', 'BF16', [1, 512])
    add('confidence_head.proj.bias', 'BF16', [1])
    header, payload = {}, bytearray()
    for name, dtype, shape in tensors:
        count = 1
        for n in shape:
            count *= n
        rng = random.Random(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little'))
        if dtype == 'BF16':
            data = bytearray()
            for _ in range(count):
                value = rng.uniform(-.08, .08)
                bits = struct.unpack('<I', struct.pack('<f', value))[0]
                data += struct.pack('<H', (bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
        elif dtype == 'U8':
            data = bytes(rng.randrange(115, 119) for _ in range(count))
        else:
            data = bytes(rng.randrange(16, 96) | (rng.randrange(2) << 7) for _ in range(count))
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload) + len(data)])
        payload.extend(data)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    (directory / 'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    print(f'DSpark fixture: {len(header)} tensors, {len(payload)} payload bytes')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/data/minimax/dspark_config.json')
    args = ap.parse_args()
    write(args.output, args.config)
