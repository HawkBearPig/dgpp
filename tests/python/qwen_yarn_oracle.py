#!/usr/bin/env python3
"""The Qwen YaRN goldens' provenance (2026-09-18).

Prints every value tests/unit/qwen_rope_scaling_test.cpp and
tests/cuda/qsa_test.cu freeze — the YaRN inverse-frequency table, the
mrope-enlarged vs native correction band, the attention mscale and the
bf16 cos/sin the vLLM cache holds — computed by the SAME vLLM code the
user's recipe runs, not by a re-derivation.

Run inside the recipe's image (a GPU is present but unused for these host
tensors):

  HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4
  #   (this box: /data/hf/hub/... — the recipe's HF_HOME)
  docker run --rm --gpus all --entrypoint python3 \
    -v "$HUB":/ckpt:ro -v "$PWD/tests/python":/w \
    vllm/vllm-openai:qwen38-flash-next /w/qwen_yarn_oracle.py \
    /ckpt/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47

(The whole models-- directory is mounted, not the snapshot alone:
config.json is a symlink into ../../blobs. The image is the digest-pinned
one in the recipe's .env.)

What it reproduces, in the recipe's own terms:

  * start.sh emits --hf-overrides '{"text_config":{"rope_parameters":
    {"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":
    262144}}}' with VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and MAX_MODEL_LEN=524288,
    and YARN_ENABLE=true / YARN_FACTOR=2.0 in .env.
  * vllm ModelConfig._update_nested MERGES that into the checkpoint's
    existing text_config.rope_parameters, so mrope_section [11,11,10]
    survives and get_rope builds an MRotaryEmbedding (not
    YaRNScalingRotaryEmbedding).
  * MRotaryEmbedding builds its cache at max_position_embeddings * 4
    (mrope.py) and YaRN's correction band is computed from that value:
    low/high = 16/24, where a non-mrope YaRN over 262144 would use 14/22.
  * mscale = yarn_get_mscale(factor) * attn_factor = 1.0693147180559945
    rides the bf16 cos/sin cache; the attention scale stays
    head_dim**-0.5 (nvidia/qsa.py).
"""

from __future__ import annotations

import json
import sys

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope

FACTOR = 2.0
ORIGINAL = 262144


def hexes(t: torch.Tensor) -> str:
    return "[" + ", ".join(hex(int(x)) for x in t.view(torch.int32)) + "]"


def main(checkpoint: str) -> int:
    raw = json.load(open(f"{checkpoint}/config.json"))
    tc = raw["text_config"]
    merged = dict(tc.get("rope_parameters") or {})
    before = dict(merged)
    merged.update(
        {"rope_type": "yarn", "factor": FACTOR, "original_max_position_embeddings": ORIGINAL}
    )
    print("checkpoint rope_parameters:", before)
    print("after the launcher's merge:  ", merged)
    assert merged.get("mrope_section"), "the merge keeps mrope_section (MRotaryEmbedding)"

    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(head_size=tc["head_dim"], max_position=tc["max_position_embeddings"],
                        rope_parameters=merged)
    print(f"class {type(rope).__name__} rotary_dim {rope.rotary_dim} base {float(rope.base)}")
    print(f"max_position_embeddings (the band's source) {rope.max_position_embeddings}")
    print(f"mscale {rope.mscale!r} f32 {hex(int(torch.tensor(rope.mscale).view(torch.int32)))}")
    inv = rope._compute_inv_freq(rope.scaling_factor)  # what the cache uses
    print("YaRN table (recipe, band 4x):", hexes(inv))
    cache = rope._compute_cos_sin_cache().to(torch.bfloat16)
    for pos in (0, 12345, 262143, 524287):
        row = cache[pos]
        cos = ", ".join(repr(float(v)) for v in row[:4])
        sin = ", ".join(repr(float(v)) for v in row[32:36])
        print(f"  bf16 cache pos {pos}: cos {cos} ; sin {sin}")

    native = {k: v for k, v in merged.items() if k not in ("mrope_section", "mrope_interleaved")}
    with set_current_vllm_config(VllmConfig()):
        plain = get_rope(head_size=tc["head_dim"], max_position=ORIGINAL, rope_parameters=native)
    print(f"class {type(plain).__name__} (no mrope_section) mscale {plain.mscale!r}")
    print("YaRN table (native band 1x):", hexes(plain._compute_inv_freq(plain.scaling_factor)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/ckpt"))
