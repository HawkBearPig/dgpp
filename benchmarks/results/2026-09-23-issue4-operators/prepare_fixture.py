#!/usr/bin/env python3
"""Reconstruct the published diagnostic token hash from existing issue evidence."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--evidence-dir", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
prompt = (args.evidence_dir / "fixture/native.prompt.utf8").read_text()
answer = (args.evidence_dir / "latest/assistant-content.utf8").read_text()
end = answer.rindex("val_") + len("val_")
forced = prompt + answer[:end]
tokens = Tokenizer.from_file(str(args.tokenizer)).encode(forced, add_special_tokens=False).ids
sha = hashlib.sha256(struct.pack("<" + "q" * len(tokens), *tokens)).hexdigest()
assert len(tokens) == 261290
assert sha == "fe207c412e6a70d70b00bbc3c5995488ed12ce964c4ac7eb90933ab61e82785d"
args.out.mkdir(parents=True, exist_ok=False)
(args.out / "forced-prefix.txt").write_text(forced)
(args.out / "forced-prefix.ids.json").write_text(json.dumps(tokens))
(args.out / "forced-request.json").write_text(json.dumps({"model": "nvidia/Qwen3.8-Flash-Next-NVFP4", "prompt": forced, "temperature": 0, "max_tokens": 1, "logprobs": 5}))
print(json.dumps({"tokens": len(tokens), "int64_le_sha256": sha}))
