"""Host-only independent checks for the capture analyzer's number formats."""
import json
from pathlib import Path

import numpy as np
import torch

import audit

torch.set_num_threads(2)
bits = np.arange(256, dtype=np.uint8)
expected = torch.from_numpy(bits).view(torch.float8_e4m3fn).float().numpy()
assert np.array_equal(audit.e4m3(bits), expected, equal_nan=True)
rng = np.random.default_rng(4)
values = rng.normal(size=(320,640)).astype(np.float32)
weight = torch.from_numpy(values).to(torch.bfloat16).float().numpy()
expected = np.empty_like(weight)
for i in range(0,320,128):
    for j in range(0,640,128):
        block = torch.from_numpy(weight[i:i+128,j:j+128])
        scale = block.abs().max()/448
        expected[i:i+128,j:j+128] = ((block/scale).to(torch.float8_e4m3fn).float()*scale).to(torch.bfloat16).float().numpy()
assert np.array_equal(audit.fp8_blocks(weight),expected)
assert np.array_equal(audit.rounded(values),torch.from_numpy(values).to(torch.bfloat16).float().numpy())
assert audit.chain_budget(audit.rounded(values),audit.rounded(values))['pass']
result = {'e4m3_decode_all_256_codes_matches_torch':True,
          'partial_block_fp8_320x640_matches_torch':True,
          'bf16_rounding_204800_values_matches_torch':True,
          'identity_chain_budget_passes':True}
(Path(__file__).resolve().parent/'host-oracle-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
