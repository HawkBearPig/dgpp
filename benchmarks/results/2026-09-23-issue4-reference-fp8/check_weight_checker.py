#!/usr/bin/env python3
"""CPU preflight of the diagnostic using independently expanded checkpoint scales."""
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open

import issue4_fp8_weights as checker

ROOT = Path(__file__).resolve().parent
checker.SNAPSHOT = Path('/home/stephen/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13')
index = json.loads((checker.SNAPSHOT / 'model.safetensors.index.json').read_text())['weight_map']
torch.set_num_threads(2)


class Samples:
    def __init__(self, shape, dtype):
        self.shape, self.dtype, self.values = shape, dtype, {}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.values[key]
        return self.values[key[0]][key[1:]]


class Module:
    def __init__(self):
        self.parameters = {
            'w13_weight': Samples((512, 640, 2560), torch.float8_e4m3fn),
            'w2_weight': Samples((512, 2560, 320), torch.float8_e4m3fn),
            'w13_weight_scale_inv': Samples((512, 10, 40), torch.float32),
            'w2_weight_scale_inv': Samples((512, 40, 5), torch.float32),
        }

    def named_parameters(self):
        return self.parameters.items()


def source(expert, projection, kind):
    key = f'model.language_model.layers.0.mlp.experts.{expert}.{projection}.{kind}'
    with safe_open(checker.SNAPSHOT / index[key], framework='pt', device='cpu') as f:
        return f.get_tensor(key)


results = []
for rank in (0, 1):
    module = Module()
    for expert in (0, 127, 255, 511):
        weights, scales = {}, {}
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            weight = source(expert, projection, 'weight')
            small = source(expert, projection, 'weight_scale_inv')
            assert small.dtype == torch.bfloat16
            # Decode BF16 bits without relying on the checker's dtype cast.
            expanded = (small.view(torch.int16).to(torch.int32) << 16).view(torch.float32)
            expanded = expanded.repeat_interleave(128, 0).repeat_interleave(128, 1)
            assert expanded.shape == weight.shape
            span = slice(rank * 320, (rank + 1) * 320)
            weights[projection] = weight[:, span] if projection == 'down_proj' else weight[span]
            expanded = expanded[:, span] if projection == 'down_proj' else expanded[span]
            rows, cols = expanded.shape
            blocks = expanded.reshape(rows // 64, 64, cols // 64, 64)
            grid = blocks[:, 0, :, 0].clone()
            assert torch.equal(blocks, grid[:, None, :, None].expand_as(blocks))
            scales[projection] = grid
        for kind, data in (('weight', weights), ('weight_scale_inv', scales)):
            # Concatenate bytes for FP8 tensors; some CPU builds lack FP8 cat.
            gate, up, down = [data[p] for p in ('gate_proj', 'up_proj', 'down_proj')]
            if kind == 'weight':
                merged = torch.cat([gate.view(torch.uint8), up.view(torch.uint8)]).view(gate.dtype)
            else:
                merged = torch.cat([gate, up])
            module.parameters['w13_' + kind].values[expert] = merged
            module.parameters['w2_' + kind].values[expert] = down
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        checker.check(module, rank, directory)
        receipt = json.loads((directory / 'fp8-weight-checks.json').read_text())
        assert receipt['all_pass'] and len(receipt['checks']) == 24
        # A real scale mismatch must still be rejected.
        module.parameters['w13_weight_scale_inv'].values[0][0, 0] *= 2
        try:
            checker.check(module, rank, directory)
        except AssertionError:
            rejected = True
        else:
            raise AssertionError('Checker accepted a deliberately corrupted scale')
    results.append(dict(rank=rank, exact_checks=len(receipt['checks']), corrupted_scale_rejected=rejected))
output = dict(cpu_only=True, checker_sha256=hashlib.sha256((ROOT / 'issue4_fp8_weights.py').read_bytes()).hexdigest(),
              independent_scale_construction='BF16 bit expansion to FP32, full 128x128 blocks, TP slice, then 64x64 regrouping',
              results=results, all_pass=True)
(ROOT / 'weight-checker-preflight.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
