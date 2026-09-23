#!/usr/bin/env python3
"""Independent checkpoint/NumPy replay of captured real MoE rows.

Products accumulate in FP64. Local stage checks use captured preceding inputs;
the complete expert-chain check uses its own intermediate values. Neither
validates the earlier layers which produced the captured MoE input.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path('/home/stephen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47')


def bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def rounded(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return bf16((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16)


def e4m3(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    exponent, fraction = (bits >> 3) & 15, bits & 7
    value = np.where(exponent == 0, fraction.astype(np.float64) * 2**-9,
                     (1 + fraction.astype(np.float64) / 8) * np.exp2(exponent.astype(np.int32) - 7))
    value = np.where((bits & 127) == 127, np.nan, value)
    return np.where(bits & 128, -value, value).astype(np.float32)


def fp8_blocks(weight):
    """Quantize the TP slice in 128x128 blocks, including partial edge blocks."""
    nr, nc = weight.shape
    padded = np.pad(weight, ((0, (-nr) % 128), (0, (-nc) % 128)))
    pr, pc = padded.shape
    blocks = padded.reshape(pr // 128, 128, pc // 128, 128)
    maximum = np.abs(blocks).max(axis=(1, 3), keepdims=True)
    scale = np.where(maximum > 0, maximum / np.float32(448), np.float32(1))
    scaled = blocks / scale
    _, exponent = np.frexp(np.abs(scaled))
    step = np.exp2(np.maximum(exponent - 4, -9)).astype(np.float32)
    code = np.clip(np.rint(scaled / step) * step, -448, 448)
    return rounded(code * scale).reshape(pr, pc)[:nr, :nc]


class Weights:
    def __init__(self):
        self.index = json.loads((CHECKPOINT / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}
        self.hashes = {}

    def tensor(self, name):
        path = CHECKPOINT / self.index[name]
        if path not in self.headers:
            with path.open('rb') as f:
                size, = struct.unpack('<Q', f.read(8))
                self.headers[path] = size + 8, json.loads(f.read(size))
        offset, header = self.headers[path]
        info = header[name]
        dtype = {'BF16': '<u2', 'F32': '<f4', 'U8': 'u1', 'F8_E4M3': 'u1'}[info['dtype']]
        data = np.memmap(path, mode='r', dtype=dtype,
                         offset=offset + info['data_offsets'][0], shape=tuple(info['shape']))
        if name not in self.hashes:
            self.hashes[name] = hashlib.sha256(data.tobytes()).hexdigest()
        return bf16(data) if info['dtype'] == 'BF16' else np.asarray(data)

    def expert(self, prefix, expert, matrix, rank, inter):
        name = prefix + f'experts.{expert}.{matrix}_proj'
        payload = self.tensor(name + '.weight')
        scales = self.tensor(name + '.weight_scale')
        scale2 = self.tensor(name + '.weight_scale_2').item()
        if matrix == 'down':
            payload = payload[:, rank * inter // 2:(rank + 1) * inter // 2]
            scales = scales[:, rank * inter // 16:(rank + 1) * inter // 16]
        else:
            payload = payload[rank * inter:(rank + 1) * inter]
            scales = scales[rank * inter:(rank + 1) * inter]
        codes = np.stack((payload & 15, payload >> 4), axis=-1).reshape(payload.shape[0], -1)
        levels = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
        values = levels[codes & 7] * np.where(codes & 8, -1., 1.)
        values *= np.repeat(e4m3(scales), 16, axis=1)
        assert np.array_equal(values, rounded(values)), 'block products must be exact BF16'
        # Checkpoint stores the multiplier; DGPP stores its F32 reciprocal.
        return values, float(np.float32(1) / np.float32(scale2))


def stats(reference, actual):
    r, a = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    error = np.abs(r - a)
    floor = .001 * np.max(np.abs(r), axis=-1, keepdims=True)
    return {'relative_l2': float(np.linalg.norm(r-a) / max(np.linalg.norm(r), 1e-30)),
            'max_abs': float(error.max()),
            'max_relative_with_row_floor': float((error / np.maximum(np.maximum(np.abs(r), floor), 1e-30)).max()),
            'nonfinite': int((~np.isfinite(r) | ~np.isfinite(a)).sum())}


def chain_budget(reference, actual):
    """Existing qwen_moe_test budget, not a fixture-selected tolerance."""
    r, a = np.asarray(reference, dtype=np.float32), np.asarray(actual, dtype=np.float32)
    rb, ab = r.view(np.uint32) >> 16, a.view(np.uint32) >> 16
    def ordered(bits):
        return np.where(bits & 32768, 32768 - (bits & 32767).astype(np.int32), bits.astype(np.int32) + 32768)
    ulps = np.abs(ordered(rb) - ordered(ab))
    rms = np.sqrt(np.mean(r.astype(np.float64)**2))
    ulps = np.where(np.abs(r.astype(np.float64)-a) <= .02*rms, 0, ulps)
    s = stats(r, a)
    s.update(hard=int((ulps > 12).sum()), soft_fraction=float((ulps > 4).mean()))
    s['pass'] = s['nonfinite'] == 0 and s['hard'] == 0 and s['soft_fraction'] < .02 and s['relative_l2'] < .004
    return s


def swiglu(gate, up):
    gate = np.asarray(gate, dtype=np.float64)
    return rounded(rounded(gate / (1 + np.exp(-gate))) * up)


def ordered_sum(down, weights):
    result = np.zeros(down.shape[1], dtype=np.float32)
    for vector, weight in zip(down, weights):
        result = (result.astype(np.float64) + vector.astype(np.float64)*float(weight)).astype(np.float32)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', type=Path, default=ROOT/'raw/captures')
    args = ap.parse_args()
    weights = Weights()
    results, folds = [], []
    for layer in range(48):
        cache = {}
        prefix = f'model.language_model.layers.{layer}.mlp.'
        router = weights.tensor(prefix+'gate.weight').astype(np.float64)
        shared_weight = weights.tensor(prefix+'shared_expert_gate.weight').astype(np.float64).reshape(-1)
        for position in (0, 260096):
            partials = {}
            for rank in (0, 1):
                path = args.captures/f'rank{rank}/position{position}/layer{layer}'
                tokens, hidden, inter, experts, topk = np.fromfile(path/'geometry.bin', dtype='<i4')
                shared_inter, shared_fp8 = np.fromfile(path/'shared_geometry.bin', dtype='<i4')
                for row in (0, int(tokens)//2, int(tokens)-1):
                    def got(name, dtype='<u2'):
                        data = np.fromfile(path/f'row{row}_{name}.bin', dtype=dtype)
                        return bf16(data) if dtype == '<u2' else data
                    x = got('input').astype(np.float64)
                    ids, router_weights = got('ids', '<i4'), got('weights', '<f4')
                    logits = got('logits', '<f4')
                    expected_ids = np.sort(np.lexsort((np.arange(experts), -logits))[:topk])
                    probabilities = np.exp(logits[ids].astype(np.float64)-float(logits.max()))
                    expected_weights = rounded(probabilities/probabilities.sum())
                    independent_logits = rounded(router@x)
                    independent_ids = np.sort(np.lexsort((np.arange(experts), -independent_logits))[:topk])
                    checks = {'router_logits': stats(independent_logits, logits),
                              'router_ids_from_independent_logits_exact': bool(np.array_equal(ids,independent_ids)),
                              'router_independent_id_difference': [int(e) for e in np.setxor1d(ids,independent_ids)],
                              'router_ids_from_captured_logits_exact': bool(np.array_equal(ids, expected_ids)),
                              'router_weights': stats(expected_weights, router_weights),
                              'segmented_source_rows_exact': bool(np.all(got('source_rows','<i4') == row))}
                    stages = {k: got(k, '<f4' if k == 'down' else '<u2').reshape(topk,-1) for k in ('gate','up','act','down')}
                    full_down = []
                    local_gates, local_ups, local_down = [], [], []
                    for slot, expert in enumerate(ids):
                        products = {}
                        for matrix in ('gate','up','down'):
                            key = rank, int(expert), matrix
                            if key not in cache: cache[key] = weights.expert(prefix, expert, matrix, rank, inter)
                            w, divisor = cache[key]
                            if matrix != 'down': products[matrix] = rounded((w@x)/divisor)
                            else:
                                local_down.append((w@stages['act'][slot].astype(np.float64))/divisor)
                                full_down.append(((w@swiglu(products['gate'],products['up']).astype(np.float64))/divisor).astype(np.float32))
                        local_gates.append(products['gate']);local_ups.append(products['up'])
                    checks.update(gate=stats(local_gates,stages['gate']), up=stats(local_ups,stages['up']),
                                  activation=stats(swiglu(stages['gate'],stages['up']), stages['act']),
                                  down_from_captured_activation=stats(local_down,stages['down']),
                                  accumulation=stats(ordered_sum(stages['down'],router_weights),got('routed','<f4')))
                    routed = ordered_sum(np.asarray(full_down),router_weights)
                    shared = {}
                    for matrix in ('gate','up','down'):
                        key = rank,'shared',matrix
                        if key not in cache:
                            w = weights.tensor(prefix+f'shared_expert.{matrix}_proj.weight')
                            w = w[:,rank*shared_inter:(rank+1)*shared_inter] if matrix == 'down' else w[rank*shared_inter:(rank+1)*shared_inter]
                            cache[key] = (fp8_blocks(w) if shared_fp8 else w).astype(np.float64)
                        w = cache[key]
                        if matrix != 'down': shared[matrix] = rounded(w@x)
                        else:
                            checks['shared_down_from_captured_activation'] = stats(w@got('shared_act').astype(np.float64),got('shared_down','<f4'))
                            shared['down'] = (w@swiglu(shared['gate'],shared['up']).astype(np.float64)).astype(np.float32)
                    checks['shared_gate'] = stats(shared['gate'],got('shared_gate'))
                    checks['shared_up'] = stats(shared['up'],got('shared_up'))
                    checks['shared_activation'] = stats(swiglu(got('shared_gate'),got('shared_up')),got('shared_act'))
                    shared_logit = float(rounded(shared_weight@x))
                    shared_gate = float(rounded(1/(1+np.exp(-shared_logit))))
                    checks['shared_weight'] = stats(np.asarray([shared_gate]),got('shared_weight','<f4'))
                    combined = rounded((routed.astype(np.float64)+shared['down'].astype(np.float64)*shared_gate).astype(np.float32))
                    checks['complete_local_chain'] = chain_budget(combined,got('local_combined'))
                    partials[rank,row] = combined, got('local_combined'), got('moe_folded')
                    results.append({'layer':layer,'rank':rank,'position':position+row,'checks':checks})
            for row in (0, int(tokens)//2, int(tokens)-1):
                a,b = partials[0,row],partials[1,row]
                folds.append({'layer':layer,'position':position+row,
                              'captured_fold_exact':bool(np.array_equal(rounded(a[1]+b[1]),a[2]) and np.array_equal(a[2],b[2])),
                              'independent_chain_fold':chain_budget(rounded(a[0]+b[0]),a[2])})
        (ROOT/'audit.json').write_text(json.dumps({'rows':results,'folds':folds},indent=2)+'\n')
        print('layer',layer,'local chain failures',sum(not r['checks']['complete_local_chain']['pass'] for r in results),flush=True)
    (ROOT/'tensor-hashes.json').write_text(json.dumps(weights.hashes,indent=2)+'\n')
    summary={'rows':len(results),'folds':len(folds),
             'local_chain_failures':sum(not r['checks']['complete_local_chain']['pass'] for r in results),
             'folded_chain_failures':sum(not r['independent_chain_fold']['pass'] for r in folds),
             'exact_fold_failures':sum(not r['captured_fold_exact'] for r in folds),
             'route_or_segment_failures':sum(not r['checks']['router_ids_from_captured_logits_exact'] or not r['checks']['segmented_source_rows_exact'] for r in results)}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)


if __name__ == '__main__': main()
