#!/usr/bin/env python3
"""Independent NumPy reference and synthetic checkpoint for MiniMax M3.1.

The fixture enumerates tensors independently of the C++ binding. It uses the
published weight layouts, FP32 routers, Gemma norms, SwiGLU-OAI and causal MSA.
The oracle uses FP64 matrix accumulation and full-precision attention softmax;
all model BF16 boundaries remain explicit. Indexed real checkpoints are mapped
on demand with a bounded decoded-weight cache for short layer-oracle probes.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import mmap
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def bf16(x):
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def rounded(x):
    return f32(bf16(x))


def fp8_table():
    result = []
    for b in range(256):
        sign = -1 if b & 128 else 1
        e, m = (b >> 3) & 15, b & 7
        value = (m / 8) * 2**-6 if e == 0 else (1 + m / 8) * 2**(e-7)
        result.append(np.nan if (b & 127) == 127 else sign * value)
    return np.array(result, dtype=np.float32)


FP8 = fp8_table()
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def fixture(path):
    path.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'tests/data/minimax/tiny_config.json').read_text())
    c = config['text_config']
    tensors = {}

    def add(name, dtype, shape, kind='weight'):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little'))
        if dtype == 'BF16':
            scale = .06 if kind == 'norm' else .3
            value = bf16(rng.normal(0, scale, shape))
        elif dtype == 'F32':
            value = np.asarray(1 + .3 * rng.random() if kind == 'global' else rng.normal(0, .04, shape), dtype='<f4')
        elif dtype == 'F8_E4M3':
            if kind == 'scale':
                value = rng.integers(8, 24, shape, dtype=np.uint8)
            else:
                value = rng.integers(0, 256, shape, dtype=np.uint8)
                value[(value & 127) == 127] &= np.uint8(0xf0)
        elif kind == 'scale':
            value = rng.integers(112, 116, shape, dtype=np.uint8)
        else:
            value = rng.integers(0, 256, shape, dtype=np.uint8)
        tensors[name] = (dtype, list(shape), value.tobytes())

    def mx(base, n, k):
        add(base+'.weight', 'F8_E4M3', (n, k))
        add(base+'.weight_scale_inv', 'U8', (n, k//32), 'scale')

    def mlp(base, i):
        mx(base+'gate_proj', i, h)
        mx(base+'up_proj', i, h)
        mx(base+'down_proj', h, i)

    h, d = c['hidden_size'], c['head_dim']
    add('language_model.model.embed_tokens.weight', 'BF16', (c['vocab_size'], h))
    add('language_model.model.norm.weight', 'BF16', (h,), 'norm')
    add('language_model.lm_head.weight', 'BF16', (c['vocab_size'], h))
    for layer in range(c['num_hidden_layers']):
        p = f'language_model.model.layers.{layer}.'
        for name in ('input_layernorm', 'post_attention_layernorm'):
            add(p+name+'.weight', 'BF16', (h,), 'norm')
        a = p+'self_attn.'
        for name, n in [('q_proj', c['num_attention_heads']*d), ('k_proj', c['num_key_value_heads']*d),
                        ('v_proj', c['num_key_value_heads']*d), ('index_q_proj', c['sparse_attention_config']['sparse_num_index_heads']*d),
                        ('index_k_proj', d)]:
            mx(a+name, n, h)
        mx(a+'o_proj', h, c['num_attention_heads']*d)
        for name in ('q_norm', 'k_norm', 'index_q_norm', 'index_k_norm'):
            add(a+name+'.weight', 'BF16', (d,), 'norm')
        if not c['moe_layer_freq'][layer]:
            mlp(p+'mlp.', c['dense_intermediate_size'])
            continue
        p += 'block_sparse_moe.'
        add(p+'gate.weight', 'F32', (c['num_local_experts'], h))
        add(p+'e_score_correction_bias', 'F32', (c['num_local_experts'],))
        mlp(p+'shared_experts.', c['shared_intermediate_size'])
        for expert in range(c['num_local_experts']):
            for name, n, k in [('w1', c['intermediate_size'], h), ('w3', c['intermediate_size'], h), ('w2', h, c['intermediate_size'])]:
                base = p+f'experts.{expert}.{name}'
                add(base+'.weight_packed', 'U8', (n, k//2))
                add(base+'.weight_scale', 'F8_E4M3', (n, k//16), 'scale')
                add(base+'.weight_global_scale', 'F32', (), 'global')
                add(base+'.input_global_scale', 'F32', (), 'global')
    header, payload, offset = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset+len(data)])
        payload.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with (path/'model.safetensors').open('wb') as out:
        out.write(struct.pack('<Q', len(raw)))
        out.write(raw)
        for data in payload:
            out.write(data)
    (path/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'Wrote {len(tensors)} tensors, {offset} bytes to {path}')


class Weights:
    def __init__(self, path):
        index = path/'model.safetensors.index.json'
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        self.header, self.sources, self.maps = {}, {}, []
        for shard in shards:
            with (path/shard).open('rb') as f:
                data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.maps.append(data)
            length, = struct.unpack('<Q', data[:8])
            header = json.loads(data[8:8+length])
            for name, value in header.items():
                if name == '__metadata__':
                    continue
                if name in self.header:
                    raise ValueError(f'duplicate tensor {name}')
                self.header[name] = value
                self.sources[name] = (data, 8+length)
        self.decoded = OrderedDict()
        self.decoded_bytes = 0
        self.cache_limit = 512 << 20

    def raw(self, name, rows=None):
        t = self.header[name]
        data, start = self.sources[name]
        dtype = {'BF16': '<u2', 'F32': '<f4', 'U8': 'u1', 'F8_E4M3': 'u1'}[t['dtype']]
        begin, end = t['data_offsets']
        x = np.frombuffer(data, dtype=dtype, count=(end-begin)//np.dtype(dtype).itemsize, offset=start+begin).reshape(t['shape'])
        if rows is not None:
            x = x[rows]
        return x

    def tensor(self, name, rows=None):
        x = self.raw(name, rows)
        dtype = self.header[name]['dtype']
        return f32(x) if dtype == 'BF16' else FP8[x] if dtype == 'F8_E4M3' else x

    def matrix(self, base):
        if base in self.decoded:
            self.decoded.move_to_end(base)
            return self.decoded[base]
        if base+'.weight' in self.header:
            matrix = self.tensor(base+'.weight')
            if base+'.weight_scale_inv' in self.header:
                scales = np.exp2(self.tensor(base+'.weight_scale_inv').astype(np.float32)-127)
                matrix = matrix * np.repeat(scales, 32, axis=1)
        else:
            codes = self.tensor(base+'.weight_packed')
            unpacked = np.stack([codes & 15, codes >> 4], axis=-1).reshape(codes.shape[0], -1)
            matrix = FP4[unpacked] * np.repeat(self.tensor(base+'.weight_scale'), 16, axis=1)
            matrix = matrix.astype(np.float64) / float(self.tensor(base+'.weight_global_scale'))
        matrix = matrix.astype(np.float64)
        while self.decoded and self.decoded_bytes + matrix.nbytes > self.cache_limit:
            _, old = self.decoded.popitem(last=False)
            self.decoded_bytes -= old.nbytes
        if matrix.nbytes <= self.cache_limit:
            self.decoded[base] = matrix
            self.decoded_bytes += matrix.nbytes
        return matrix


def norm(x, weight, eps):
    variance = np.mean(x.astype(np.float64)**2, axis=-1, keepdims=True).astype(np.float32)
    rstd = (1 / np.sqrt(variance + np.float32(eps))).astype(np.float32)
    return rounded(x * rstd * (1+weight))


def rotary(x, positions, c):
    d = c['rotary_dim']
    inv = (1/np.power(float(c['rope_theta']), np.arange(0, d, 2, dtype=np.float64)/d)).astype(np.float32)
    angles = positions.astype(np.float32)[:, None] * inv[None, :]
    cs, sn = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    lo, hi = x[..., :d//2].copy(), x[..., d//2:d].copy()
    result = x.copy()
    result[..., :d//2] = rounded(lo*cs-hi*sn)
    result[..., d//2:d] = rounded(hi*cs+lo*sn)
    return result


def activate(g, u, c):
    g = np.minimum(g, c['swiglu_limit'])
    u = np.clip(u, -c['swiglu_limit'], c['swiglu_limit'])
    return rounded((g/(1+np.exp(-np.float32(c['swiglu_alpha'])*g)))*(u+1))


def column_parts(x, matrix, world, round_output=True):
    """TP row projections round locally before the BF16 boundary fold."""
    width = matrix.shape[1] // world
    parts = []
    for rank in range(world):
        columns = slice(rank*width, (rank+1)*width)
        y = x[:, columns].astype(np.float64) @ matrix[:, columns].T
        parts.append(rounded(y) if round_output else y.astype(np.float32))
    return np.stack(parts)


def fold(parts):
    total = np.zeros_like(parts[0], dtype=np.float32)
    for part in parts:
        total += part
    return rounded(total)


def mlp(x, base, w, c, expert=False, round_output=True, world=1):
    gate, up, down = ('w1','w3','w2') if expert else ('gate_proj','up_proj','down_proj')
    g = rounded(x.astype(np.float64) @ w.matrix(base+gate).T)
    u = rounded(x.astype(np.float64) @ w.matrix(base+up).T)
    return column_parts(activate(g,u,c), w.matrix(base+down), world, round_output)


def attention_forward(n, layer, w, c, world=1):
    t = n.shape[0]
    qh, kh, d = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
    ih = c['sparse_attention_config']['sparse_num_index_heads']
    topk = c['sparse_attention_config']['sparse_topk_blocks']
    p = f'language_model.model.layers.{layer}.'
    a = p+'self_attn.'
    def proj(name, heads, normalized=True):
        y = rounded(n.astype(np.float64) @ w.matrix(a+name+'_proj').T).reshape(t, heads, d)
        if normalized:
            y = norm(y, w.tensor(a+name+'_norm.weight'), c['rms_norm_eps'])
            y = rotary(y, np.arange(t), c)
        return y
    q, k, v = proj('q', qh), proj('k', kh), proj('v', kh, False)
    iq, ik = proj('index_q', ih), proj('index_k', 1)
    attention = np.zeros_like(q)
    for row in range(t):
        for index in range(ih):
            scores = ik[:row+1,0].astype(np.float64) @ iq[row,index].astype(np.float64)
            blocks = (row+128)//128
            maximum = np.array([scores[b*128:min((b+1)*128,row+1)].max() for b in range(blocks)])
            maximum[-1] = np.inf
            selected = sorted(sorted(range(blocks), key=lambda b: (-maximum[b], b))[:topk])
            tokens = np.concatenate([np.arange(b*128,min((b+1)*128,row+1)) for b in selected])
            for head in range(index*(qh//ih), (index+1)*(qh//ih)):
                kv = head//(qh//kh)
                score = (k[tokens,kv].astype(np.float64) @ q[row,head].astype(np.float64)) / np.sqrt(d)
                prob = np.exp(score-score.max())
                attention[row,head] = rounded((prob @ v[tokens,kv].astype(np.float64))/prob.sum())
    return fold(column_parts(attention.reshape(t,-1), w.matrix(a+'o_proj'), world))


def ffn_forward(n, layer, w, c, world=1):
    t, h = n.shape
    p = f'language_model.model.layers.{layer}.'
    routes = None
    if not c['moe_layer_freq'][layer]:
        y = fold(mlp(n,p+'mlp.',w,c,world=world))
    else:
        p += 'block_sparse_moe.'
        logits = (n.astype(np.float64) @ w.matrix(p+'gate').T).astype(np.float32)
        scores = 1/(1+np.exp(-logits))
        biased = scores+w.tensor(p+'e_score_correction_bias')
        routes = np.array([sorted(sorted(range(c['num_local_experts']),key=lambda e:(-row[e],e))[:c['num_experts_per_tok']]) for row in biased])
        picked = np.take_along_axis(scores,routes,axis=1)
        weights = picked/np.maximum(picked.sum(axis=1,keepdims=True),1e-20)*c['routed_scaling_factor']
        y = np.zeros((world,t,h),dtype=np.float64)
        for e in range(c['num_local_experts']):
            rows, slots = np.where(routes==e)
            if len(rows):
                y[:,rows] += weights[rows,slots,None]*mlp(n[rows],p+f'experts.{e}.',w,c,True,False,world).astype(np.float64)
        y = fold(rounded(rounded(y)+mlp(n,p+'shared_experts.',w,c,world=world)))
    return y, routes


def layer_forward(x, layer, w, c, world=1):
    p = f'language_model.model.layers.{layer}.'
    n = norm(x, w.tensor(p+'input_layernorm.weight'), c['rms_norm_eps'])
    x = rounded(x + attention_forward(n, layer, w, c, world))
    n = norm(x, w.tensor(p+'post_attention_layernorm.weight'), c['rms_norm_eps'])
    y, routes = ffn_forward(n, layer, w, c, world)
    return rounded(x+y), routes


def relative_rms(actual, expected):
    return np.linalg.norm(actual.astype(np.float64)-expected) / max(np.linalg.norm(expected.astype(np.float64)), 1e-20)


def check_stages(engine, x, layer, w, c, world):
    """Isolate each operation using its recorded input; this is not an end-to-end quality gate."""
    p = f'language_model.model.layers.{layer}.'
    def compare(stage, expected):
        actual = f32(np.fromfile(engine/f'{stage}-{layer}.bf16', dtype='<u2').reshape(x.shape))
        rms = relative_rms(actual, expected)
        print(f'Layer {layer} {stage}: relative RMS {rms:.6g}', flush=True)
        if not np.isfinite(rms) or rms > .01:
            raise AssertionError(f'Layer {layer} {stage}: relative RMS {rms} exceeds 1%')
        return actual
    n = compare('input_norm', norm(x, w.tensor(p+'input_layernorm.weight'), c['rms_norm_eps']))
    a = compare('attention', attention_forward(n, layer, w, c, world))
    n = compare('ffn_norm', norm(rounded(x+a), w.tensor(p+'post_attention_layernorm.weight'), c['rms_norm_eps']))
    expected, routes = ffn_forward(n, layer, w, c, world)
    y = compare('ffn', expected)
    return rounded(rounded(x+a)+y), routes


def check(path, engine, stages=False):
    c = json.loads((path/'config.json').read_text())['text_config']
    ranks = []
    if not (engine/'metadata.json').exists():
        ranks = sorted(engine.glob('rank[0-9]*'), key=lambda p:int(p.name[4:]))
        if not ranks:
            raise ValueError('no engine output metadata')
        engine = ranks[0]
    meta = json.loads((engine/'metadata.json').read_text())
    if ranks and len(ranks) != meta['world']:
        raise ValueError('incomplete vocabulary rank set')
    tokens = np.array(meta['tokens'])
    w = Weights(path)
    x = w.tensor('language_model.model.embed_tokens.weight', tokens)
    for layer in range(c['num_hidden_layers']):
        if stages:
            expected, routes = check_stages(engine, x, layer, w, c, meta.get('world',1))
        else:
            expected, routes = layer_forward(x,layer,w,c,world=meta.get('world',1))
        actual = f32(np.fromfile(engine/f'layer-{layer}.bf16',dtype='<u2').reshape(expected.shape))
        rms = np.sqrt(np.mean((actual.astype(np.float64)-expected)**2)) / max(np.sqrt(np.mean(expected.astype(np.float64)**2)),1e-20)
        print(f'Layer {layer}: teacher-forced relative RMS {rms:.6g}', flush=True)
        if not np.isfinite(rms) or rms > (.0 if stages else .01):
            raise AssertionError(f'Layer {layer}: residual composition mismatch' if stages else f'Layer {layer}: relative RMS {rms} exceeds 1%')
        if routes is not None:
            observed = np.fromfile(engine/f'routes-{layer}.i32',dtype='<i4').reshape(routes.shape)
            flips = np.mean(observed!=routes)
            print(f'  router slot disagreement: {flips:.6g}')
            if flips != 0:
                raise AssertionError(f'Layer {layer}: fixture routing mismatch; investigate ties before accepting')
        x = actual  # independent layer math, every layer sees the engine's input
    expected = norm(x,w.tensor('language_model.model.norm.weight'),c['rms_norm_eps']).astype(np.float64) @ w.matrix('language_model.lm_head').T
    if ranks:
        shards, begin = [], 0
        for rank, folder in enumerate(ranks):
            m = json.loads((folder/'metadata.json').read_text())
            if m['rank'] != rank or m['world'] != len(ranks) or m['tokens'] != meta['tokens'] or m['vocab_begin'] != begin:
                raise ValueError('rank metadata disagreement or vocabulary hole')
            for layer in range(c['num_hidden_layers']):
                if (folder/f'layer-{layer}.bf16').read_bytes() != (engine/f'layer-{layer}.bf16').read_bytes():
                    raise ValueError(f'rank {rank} hidden state disagreement at layer {layer}')
            shards.append(np.fromfile(folder/'logits.f32',dtype='<f4').reshape(len(tokens),m['vocab_count']))
            begin += m['vocab_count']
        if begin != c['vocab_size']:
            raise ValueError('incomplete vocabulary coverage')
        actual = np.concatenate(shards, axis=1)
    else:
        actual = np.fromfile(engine/'logits.f32',dtype='<f4').reshape(expected.shape)
    rms = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    print(f'Output head relative RMS {rms:.6g}')
    if rms > .01 or not np.array_equal(actual.argmax(axis=1),expected.argmax(axis=1)):
        raise AssertionError('Output head mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--checkpoint-dir',type=Path)
    parser.add_argument('--check-engine-dir',type=Path)
    parser.add_argument('--stages',action='store_true',help='Check recorded sublayer inputs independently; full-model NLL validation is still required')
    args = parser.parse_args()
    if args.fixture:
        fixture(args.fixture)
    elif args.checkpoint_dir and args.check_engine_dir:
        check(args.checkpoint_dir,args.check_engine_dir,args.stages)
    else:
        parser.error('use --fixture or --checkpoint-dir with --check-engine-dir')

if __name__ == '__main__':
    main()
