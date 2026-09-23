"""Read-only check of actual TP2 FP8 expert weights and refined block scales."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

SNAPSHOT = Path('/hf-model/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13')


def raw(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def check(module, rank, directory):
    index = json.loads((SNAPSHOT / 'model.safetensors.index.json').read_text())['weight_map']
    parameters = list(module.named_parameters())
    selected = {}
    for suffix in ('w13_weight', 'w2_weight', 'w13_weight_scale_inv', 'w2_weight_scale_inv'):
        matches = [(name, p) for name, p in parameters if name.split('.')[-1] == suffix]
        assert len(matches) == 1, (suffix, [name for name, _ in parameters])
        selected[suffix] = matches[0][1]
    assert tuple(selected['w13_weight'].shape) == (512, 640, 2560)
    assert tuple(selected['w2_weight'].shape) == (512, 2560, 320)
    assert tuple(selected['w13_weight_scale_inv'].shape) == (512, 10, 40)
    assert tuple(selected['w2_weight_scale_inv'].shape) == (512, 40, 5)
    assert rank in (0, 1)
    results = []
    for expert in (0, 127, 255, 511):
        for projection, runtime, shard in (('gate_proj', 'w13_weight', 0),
                                           ('up_proj', 'w13_weight', 1),
                                           ('down_proj', 'w2_weight', None)):
            prefix = f'model.language_model.layers.0.mlp.experts.{expert}.{projection}.'
            source = {}
            for kind in ('weight', 'weight_scale_inv'):
                key = prefix + kind
                with safe_open(SNAPSHOT / index[key], framework='pt', device='cpu') as f:
                    source[kind] = f.get_tensor(key)
            # Select raw FP8 bytes, avoiding any conversion of weight values.
            weight = source['weight'].view(torch.uint8)
            scales = source['weight_scale_inv']
            assert scales.dtype == torch.float32
            # Independently enumerate the finer grid's parent block.
            rows = torch.arange(scales.shape[0] * 2) // 2
            cols = torch.arange(scales.shape[1] * 2) // 2
            refined = scales[rows[:, None], cols[None, :]]
            if shard is not None:
                expected_weight = weight[rank * 320:(rank + 1) * 320]
                expected_scale = refined[rank * 5:(rank + 1) * 5]
                actual_weight = selected[runtime][expert, shard * 320:(shard + 1) * 320].detach().cpu().view(torch.uint8)
                actual_scale = selected['w13_weight_scale_inv'][expert, shard * 5:(shard + 1) * 5].detach().cpu()
            else:
                expected_weight = weight[:, rank * 320:(rank + 1) * 320]
                expected_scale = refined[:, rank * 5:(rank + 1) * 5]
                actual_weight = selected[runtime][expert].detach().cpu().view(torch.uint8)
                actual_scale = selected['w2_weight_scale_inv'][expert].detach().cpu()
            for kind, actual, expected in (('weight', actual_weight, expected_weight),
                                            ('weight_scale_inv', actual_scale, expected_scale)):
                actual_bytes, expected_bytes = raw(actual), raw(expected)
                row = dict(expert=expert, projection=projection, kind=kind,
                           shape=list(actual.shape), bytes=len(actual_bytes),
                           sha256=hashlib.sha256(actual_bytes).hexdigest(),
                           exact=actual.shape == expected.shape and actual_bytes == expected_bytes)
                assert row['exact'], row
                results.append(row)
    receipt = dict(rank=rank, checkpoint_block=[128, 128], runtime_block=[64, 64],
                   checks=results, all_pass=True)
    (directory / 'fp8-weight-checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Issue4: FP8 checkpoint weights and refined scales verified, rank', rank, flush=True)
