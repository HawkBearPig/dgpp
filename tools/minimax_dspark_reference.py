#!/usr/bin/env python3
"""Independent FP64 DSpark fixture oracle, including target input taps and Markov walk."""
import argparse
import json
from pathlib import Path
import numpy as np
from minimax_reference_dump import Weights, rounded, f32, norm, rotary, activate, relative_rms


def check(path, dump):
    c = json.loads((path / 'dspark/config.json').read_text())['text_config']
    w, target = Weights(path / 'dspark'), Weights(path)
    p = 'language_model.model.dspark.'
    ids = np.fromfile(dump / 'tokens.i64', dtype='<i8')
    T, H, D, QH, KH = len(ids), c['hidden_size'], c['head_dim'], c['num_attention_heads'], c['num_key_value_heads']
    taps = c['dflash_config']['target_layer_ids']
    outputs = []
    for layer in taps:
        if layer == 0:
            outputs.append(target.tensor('language_model.model.embed_tokens.weight')[ids])
        else:
            outputs.append(f32(np.fromfile(dump / f'target-layer-{layer - 1}.bf16', dtype='<u2')).reshape(T, H))
    features = np.concatenate(outputs, axis=1).astype(np.float64) @ w.tensor(p + 'fc.weight').astype(np.float64).T
    got_features = np.fromfile(dump / 'features.f32', dtype='<f4').reshape(T, H)
    error = relative_rms(got_features, features)
    if error > 2e-6:
        raise AssertionError(f'target input taps/FC mismatch {error}')
    print(f'target input taps and FC relative RMS {error:.3g}')
    context = norm(rounded(features), w.tensor(p + 'hidden_norm.weight'), c['rms_norm_eps'])
    def compare(name, expected, threshold=.004):
        actual = f32(np.fromfile(dump / (name + '.bf16'), dtype='<u2')).reshape(expected.shape)
        error = relative_rms(actual, expected)
        print(f'{name}: relative RMS {error:.6g}')
        if not np.isfinite(error) or error > threshold:
            raise AssertionError(f'{name}: {error} > {threshold}')
        return actual
    compare('context', context, .0005)
    R = c['dflash_config']['block_size'] + 1
    tokens = [7] + [c['dspark_noise_token_id']] * (R - 1)
    residual = target.tensor('language_model.model.embed_tokens.weight')[tokens]
    pending = None
    pos = np.arange(T, T + R)
    def linear(x, base):
        return rounded(x.astype(np.float64) @ w.matrix(base).T)
    def qk(x, name, positions, heads):
        x = x.reshape(len(x), heads, D)
        x = norm(x, w.tensor(name + '.weight'), c['rms_norm_eps'])
        return rotary(x, positions, c)
    for layer in range(c['num_hidden_layers']):
        lp = p + f'layers.{layer}.decoder_layer.'
        x = residual if pending is None else residual.astype(np.float32) + pending
        residual = rounded(x)
        n = norm(x, w.tensor(lp + 'input_layernorm.weight'), c['rms_norm_eps'])
        q = qk(linear(n, lp + 'self_attn.q_proj'), lp + 'self_attn.q_norm', pos, QH)
        k = qk(linear(n, lp + 'self_attn.k_proj'), lp + 'self_attn.k_norm', pos, KH)
        v = linear(n, lp + 'self_attn.v_proj').reshape(R, KH, D)
        ck = qk(linear(context, lp + 'self_attn.k_proj'), lp + 'self_attn.k_norm', np.arange(T), KH)
        cv = linear(context, lp + 'self_attn.v_proj').reshape(T, KH, D)
        keys, values = np.concatenate([ck, k]), np.concatenate([cv, v])
        compare(f'k-{layer}', keys)
        compare(f'v-{layer}', values)
        output = np.zeros_like(q)
        for r in range(R):
            lo = max(0, int(pos[r]) - c['sliding_window'] + 1)
            for h in range(QH):
                kh = h // (QH // KH)
                score = q[r, h].astype(np.float64) @ keys[lo:, kh].astype(np.float64).T / np.sqrt(D)
                prob = np.exp(score - score.max()); prob /= prob.sum()
                output[r, h] = rounded(prob @ values[lo:, kh].astype(np.float64))
        attention = linear(output.reshape(R, -1), lp + 'self_attn.o_proj')
        x = residual.astype(np.float32) + attention
        residual = rounded(x)
        n = norm(x, w.tensor(lp + 'post_attention_layernorm.weight'), c['rms_norm_eps'])
        gate, up = linear(n, lp + 'mlp.gate_proj'), linear(n, lp + 'mlp.up_proj')
        pending = linear(activate(gate, up, c), lp + 'mlp.down_proj')
    hidden = norm(residual.astype(np.float32) + pending, w.tensor(p + 'final_norm.weight'), c['rms_norm_eps'])
    actual_hidden = compare('hidden', hidden)
    # Isolate head and selector checks at the observed BF16 hidden boundary:
    # tiny permitted backbone rounding must not hide a wrong predecessor.
    logits = actual_hidden.astype(np.float64) @ target.tensor('language_model.lm_head.weight').astype(np.float64).T
    actual_logits = np.fromfile(dump / 'logits.f32', dtype='<f4').reshape(R, c['vocab_size'])
    error = relative_rms(actual_logits, logits)
    if error > 2e-6:
        raise AssertionError(f'head mismatch {error}')
    pred = w.tensor(p + 'markov_head.markov_w1.weight').astype(np.float64)
    succ = w.tensor(p + 'markov_head.markov_w2.weight').astype(np.float64)
    proposals = np.fromfile(dump / 'proposals.i32', dtype='<i4')
    confidence = np.fromfile(dump / 'confidence.f32', dtype='<f4')
    conf_w = w.tensor(p + 'confidence_head.proj.weight').astype(np.float64).ravel()
    conf_b = float(w.tensor(p + 'confidence_head.proj.bias')[0])
    previous = 7
    for r in range(R - 1):
        candidates = np.lexsort((np.arange(c['vocab_size']), -actual_logits[r + 1]))[:16]
        corrected = actual_logits[r + 1, candidates].astype(np.float64) + succ[candidates] @ pred[previous]
        chosen = candidates[np.lexsort((candidates, -corrected))[0]]
        if chosen != proposals[r]:
            raise AssertionError(f'vanilla Markov selection at {r}: {chosen} != {proposals[r]}')
        expected = 1 / (1 + np.exp(-(np.concatenate([actual_hidden[r + 1], pred[previous]]) @ conf_w + conf_b)))
        if abs(expected - confidence[r]) > 2e-6:
            raise AssertionError(f'learned confidence at {r}: {expected} != {confidence[r]}')
        previous = int(proposals[r])
    print('head, vanilla Markov walk and learned confidence: pass')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--engine-dir', type=Path, required=True)
    a = ap.parse_args()
    check(a.checkpoint_dir, a.engine_dir)
