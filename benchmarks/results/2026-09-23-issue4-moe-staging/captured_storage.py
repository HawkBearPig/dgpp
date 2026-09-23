#!/usr/bin/env python3
"""Compare explicit reference BF16 storage boundaries on frozen captured inputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent/'2026-09-23-issue4-moe-audit'
spec = importlib.util.spec_from_file_location('moe_audit', AUDIT/'audit.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
results=[]
for rank in (0,1):
 for position in (0,260096):
  for layer in range(48):
   path=AUDIT/f'raw/captures/rank{rank}/position{position}/layer{layer}'
   tokens, hidden, inter, experts, topk=np.fromfile(path/'geometry.bin',dtype='<i4')
   for row in (0,int(tokens)//2,int(tokens)-1):
    def got(name): return np.fromfile(path/f'row{row}_{name}.bin',dtype='<f4')
    down=got('down').reshape(topk,hidden); weights=got('weights')
    original=a.ordered_sum(down,weights)
    weighted=a.rounded(down.astype(np.float64)*weights[:,None])
    staged=a.rounded(weighted.astype(np.float64).sum(axis=0))
    results.append(dict(rank=rank,position=position,layer=layer,row=row,
      weighted_changed=int(np.count_nonzero(weighted != down*weights[:,None])),
      rounded_outputs_changed=int(np.count_nonzero(staged != a.rounded(original))),
      values=int(hidden),stats=a.stats(staged,a.rounded(original))))
report={'scope':'Frozen captured down projections and routing weights; independent FP64 products/sum with explicit BF16 stores. Not a complete layer or model replay.',
 'cases':len(results),'changed_outputs':sum(x['rounded_outputs_changed'] for x in results),
 'total_outputs':sum(x['values'] for x in results),'max_relative_l2':max(x['stats']['relative_l2'] for x in results),
 'results':results}
(ROOT/'captured-storage.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='results'})
