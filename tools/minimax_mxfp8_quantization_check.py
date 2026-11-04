#!/usr/bin/env python3
"""Compare DGPP's MXFP8 codes/scales with the pinned SGLang GPU producer.

The CUDA test first checks its own outputs against an independent host
reference and FP64 GEMM bounds. This then runs the actual upstream Triton
quantizer on the dumped inputs. Requires an otherwise idle GPU.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


SOURCE = 'python/sglang/kernels/ops/quantization/mxfp8_swizzled_triton.py'
SHA256 = '95859f64d9766e898bfcf9dc7c2712a17720a04861e8d62a97a9681ccc9c1cc3'


def check(app, source, output):
    path = source / SOURCE
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('expected the SGLang 58484246 MXFP8 producer')
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([str(app.absolute()), '--dump-quant', str(output.absolute())], check=True)
    spec = importlib.util.spec_from_file_location('_minimax_upstream_mxfp8', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cases = []
    for metadata in sorted(output.glob('*.json')):
        shape = json.loads(metadata.read_text())
        rows, k, stride = (shape[name] for name in ('rows', 'k', 'stride'))
        base = metadata.with_suffix('')
        inputs = np.fromfile(str(base) + '.input.bf16', dtype=np.uint16).reshape(rows, stride)
        x = torch.from_numpy(inputs.copy()).view(torch.bfloat16).cuda()[:, :k]
        codes, scales = module.mxfp8_quantize_swizzled(x)
        expected_codes = codes.view(torch.uint8).cpu().numpy()
        r = torch.arange(rows, device=x.device)[:, None]
        c = torch.arange(k // 32, device=x.device)[None, :]
        # The upstream output uses cuBLASLt's SWIZZLE_32_4_4 scale layout.
        blocks = (k // 32 + 3) // 4
        offsets = ((r // 128) * blocks + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
        expected_scales = scales[offsets].cpu().numpy()
        actual_codes = np.fromfile(str(base) + '.codes.u8', dtype=np.uint8).reshape(rows, k)
        actual_scales = np.fromfile(str(base) + '.scales.u8', dtype=np.uint8).reshape(rows, k // 32)
        code_errors = int(np.count_nonzero(actual_codes != expected_codes))
        scale_errors = int(np.count_nonzero(actual_scales != expected_scales))
        cases.append(dict(**shape, code_errors=code_errors, scale_errors=scale_errors))
        print(f'{rows}x{k}: code differences={code_errors}, scale differences={scale_errors}', flush=True)
    report = dict(sglang_revision='58484246d26523eaf1a7f16ab524deee288d2120',
                  source_sha256=SHA256, torch_version=torch.__version__,
                  app_sha256=hashlib.sha256(app.read_bytes()).hexdigest(), cases=cases)
    report['ok'] = len(cases) == 2 and all(not c['code_errors'] and not c['scale_errors'] for c in cases)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['ok']:
        raise RuntimeError('MXFP8 upstream quantization differs; inspect comparison.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--sglang-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.app, args.sglang_source, args.out)


if __name__ == '__main__':
    main()
