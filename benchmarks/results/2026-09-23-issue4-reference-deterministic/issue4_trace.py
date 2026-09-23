"""Temporary read-only trace of repeated issue #4 requests in pinned vLLM.

Copies values to CPU without changing model arithmetic. Synchronization can
affect a race: response parity must be checked before interpreting this trace.
"""
import hashlib
import json
from pathlib import Path

import torch

PREFIX = [248045, 8678, 198, 84642, 4581, 1328, 55616, 28203,
          494, 279, 16713, 45366, 13, 3301, 4566, 1132]
request = 0
position = -1
layer = -1
active = False
directory = None


def begin(input_ids, positions):
    global request, position, layer, active, directory
    active = False
    if input_ids is None or input_ids.numel() == 0:
        return
    tokens = input_ids.detach().reshape(-1).cpu()
    pos = positions.detach().reshape(-1, tokens.numel())[0].cpu()
    position = int(pos[0])
    if position == 0:
        if tokens.numel() < len(PREFIX) or tokens[:len(PREFIX)].tolist() != PREFIX:
            return  # Ignore synthetic profiling/warmup inputs.
        request += 1
    if request == 0 or position >= 261320:
        return
    active = True
    layer = -1
    rank = torch.distributed.get_rank()
    directory = Path('/capture') / f'rank{rank}' / f'request{request}' / f'position{position}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'input.json').write_text(json.dumps({
        'position': position, 'tokens': tokens.tolist(), 'positions': pos.tolist(),
    }) + '\n')


def set_layer(index):
    global layer
    layer = index


def capture(name, tensor):
    if not active or tensor is None:
        return
    original_shape = list(tensor.shape)
    # Full first chunk to expose early differences; final row of later chunks.
    sample = tensor if position in (0, 2048) and layer < 4 else tensor[-1:]
    host = sample.detach().contiguous().cpu()
    data = host.view(torch.uint8).numpy().tobytes()
    stem = f'layer{layer}_{name}'
    path = directory / (stem + '.bin')
    with path.open('xb') as output:
        output.write(data)
    (directory / (stem + '.json')).write_text(json.dumps({
        'dtype': str(tensor.dtype), 'original_shape': original_shape,
        'saved_shape': list(host.shape), 'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }) + '\n')


def probe_topk(logits, lengths, original, op, workspace, k, columns, row_start):
    if not active or layer != 3 or position not in (0, 2048):
        return
    n = lengths.detach().cpu()
    width = int(n.max())
    host = logits[:, :width].detach().contiguous().cpu()
    host.masked_fill_(torch.arange(width)[None, :] >= n[:, None], 0)
    outputs = [original.detach().cpu()]
    for _ in range(2):
        repeat = torch.empty_like(original)
        op(logits, lengths, repeat, workspace, k, columns)
        outputs.append(repeat.detach().cpu())
    torch.save({'logits': host, 'lengths': n, 'outputs': outputs},
               directory / f'qsa_topk_{row_start}.pt')
