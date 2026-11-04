#!/usr/bin/env python3
"""Run real MiniMax vision numerical gates on an idle GPU; outputs stay untracked.

The set covers the minimum canvas, rectangular merge order, the 128-query
tile boundary, the processor's 576-token budget and the encoder's 1024-token
allocation ceiling. Each native run precedes its independent CUDA oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--binary', type=Path, default=Path('build-release/minimax_vision_check'))
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--media', choices=['image', 'video', 'all'], default='all')
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    binary = a.binary.resolve()
    oracle = Path(__file__).with_name('minimax_vision_torch_reference.py')
    images = [(28, 28, 'noise'), (56, 56, 'gradient'), (84, 56, 'gradient'),
             (196, 196, 'gradient'), (112, 224, 'checker'), (812, 392, 'noise'),
             (672, 672, 'gradient'), (896, 896, 'checker')]
    cases = [(w, h, pattern, 1) for w, h, pattern in images] if a.media != 'video' else []
    if a.media != 'image':
        cases += [(84, 56, 'noise', 2), (196, 196, 'gradient', 4), (112, 224, 'checker', 6),
                  (84, 56, 'gradient', 8), (672, 672, 'gradient', 8), (700, 700, 'gradient', 8),
                  (896, 672, 'noise', 8), (896, 896, 'checker', 8)]
    manifest = {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'])),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'checkpoint': str(a.checkpoint_dir.resolve()), 'cases': []}
    failed = False
    for width, height, pattern, frames in cases:
        name = f'{width}x{height}-{pattern}' + (f'-{frames}frames' if frames > 1 else '')
        folder = a.output_dir / name
        folder.mkdir(exist_ok=True)
        native = [str(binary), str(a.checkpoint_dir), str(width), str(height),
                  str(folder / 'output.bf16'), str(folder), '-',
                  r'rope|patches|patch_projection|pre_norm|layer[0-9]+|projected_patches|image_rows', str(frames)]
        reference = [sys.executable, str(oracle), '--checkpoint-dir', str(a.checkpoint_dir),
                     '--engine-dir', str(folder), '--width', str(width), '--height', str(height),
                     '--frames', str(frames)]
        if pattern != 'gradient':
            if pattern == 'noise':
                rng = random.Random(1471)
                pixels = bytes(rng.randrange(256) for _ in range(width * height * 3 * frames))
            else:
                pixels = bytes(component for f in range(frames) for y in range(height) for x in range(width)
                               for component in ((220, 10, 40) if (x // 17 + y // 23 + f) % 2
                                                 else (15, 90, 240)))
            rgb = folder / 'input.rgb'
            rgb.write_bytes(pixels)
            native[6] = str(rgb)
            reference += ['--input-rgb', str(rgb)]
        with (folder / 'native.log').open('w') as log:
            nc = subprocess.run(native, stdout=log, stderr=subprocess.STDOUT).returncode
        rc = None
        if nc == 0:
            with (folder / 'reference.log').open('w') as log:
                rc = subprocess.run(reference, stdout=log, stderr=subprocess.STDOUT).returncode
        passed = nc == 0 and rc == 0
        manifest['cases'].append({'name': name, 'native_exit': nc, 'reference_exit': rc, 'pass': passed})
        (a.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{name}: {"PASS" if passed else "FAIL"}', flush=True)
        failed |= not passed
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
