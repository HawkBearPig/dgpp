#!/usr/bin/env python3
"""Locate DSpark arithmetic differences using identical captured operation inputs."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from minimax_dspark_audit import DraftReference, PREFIX, load, error
from minimax_reference_dump import Weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--sglang-source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument(
        "--projection-backend",
        choices=("cublas", "triton-native"),
        default="triton-native",
    )
    ap.add_argument(
        "--elementwise-backend",
        choices=("torch", "triton-fixed"),
        default="triton-fixed",
    )
    args = ap.parse_args()
    ref = DraftReference(
        args.checkpoint_dir / "dspark",
        args.world,
        args.sglang_source,
        native_cache=False,
        projection_backend=args.projection_backend,
        elementwise_backend=args.elementwise_backend,
    )
    target = Weights(args.checkpoint_dir)
    embed = (
        torch.from_numpy(
            np.array(target.raw("language_model.model.embed_tokens.weight"), copy=True)
        )
        .view(torch.bfloat16)
        .cuda()
    )
    c = ref.c
    H = c["hidden_size"]
    D = c["head_dim"]
    QH = c["num_attention_heads"]
    KH = c["num_key_value_heads"]
    Q = QH * D
    K = KH * D
    R = c["dflash_config"]["block_size"]
    I = c["dense_intermediate_size"]
    W = args.world
    reports = []
    for case in sorted(args.engine_dir.iterdir()):
        if not (case / "rank0").is_dir():
            continue
        roots = [case / f"rank{r}" for r in range(W)]
        blocks = sorted(
            roots[0].glob("block-*"), key=lambda p: int(p.name.split("-")[1])
        )[: args.blocks]
        for block in blocks:
            meta = json.loads((block / "metadata.json").read_text())
            pos = meta["position"]
            residual = embed[[meta["anchor"]] + [c["dspark_noise_token_id"]] * (R - 1)]
            pending = None
            rows = []

            def stages(layer, name, width, sharded=False):
                values = [
                    load(
                        root / block.name / f"stage-{layer}-{name}.bf16",
                        np.uint16,
                        (R, width),
                    )
                    for root in roots
                ]
                return torch.cat(values, -1) if sharded else values

            def check(layer, name, actual, expected):
                row = dict(
                    layer=layer,
                    stage=name,
                    relative_rms=error(actual, expected),
                    changed=int((actual != expected).sum()),
                    elements=actual.numel(),
                    row_rms=[
                        error(a, b)
                        for a, b in zip(actual.reshape(R, -1), expected.reshape(R, -1))
                    ],
                )
                rows.append(row)
                print(
                    case.name,
                    block.name,
                    layer,
                    name,
                    row["relative_rms"],
                    row["changed"],
                    "/",
                    row["elements"],
                    flush=True,
                )

            for layer in range(c["num_hidden_layers"]):
                base = PREFIX + f"layers.{layer}.decoder_layer."
                a = base + "self_attn."
                if pending is None:
                    n = ref.norm(residual, base + "input_layernorm")
                else:
                    residual, n = ref.add_norm(
                        residual, pending, base + "input_layernorm"
                    )
                actual_n = stages(layer, "input_norm", H)[0]
                check(layer, "input_norm", actual_n, n)
                qkv_parts = stages(layer, "qkv", (Q + 2 * K) // W)
                split = [p.split((Q // W, K // W, K // W), -1) for p in qkv_parts]
                actual_qkv = [torch.cat([p[j] for p in split], -1) for j in range(3)]
                for j, name in enumerate(("q", "k", "v")):
                    expected = ref.linear(actual_n, a + name + "_proj")
                    check(layer, name + "_proj", actual_qkv[j], expected)
                actual_q = stages(layer, "q", Q // W, True).reshape(R, QH, D)
                expected_q = ref.qk(actual_qkv[0], a + "q_norm", pos, QH)
                check(layer, "q_norm_rope", actual_q, expected_q)
                keys, values = tuple(
                    torch.cat(
                        [
                            load(
                                root / block.name / f"{name}-{layer}.bf16",
                                np.uint16,
                                (pos + R, KH // W, D),
                            )
                            for root in roots
                        ],
                        1,
                    )
                    for name in ("k", "v")
                )
                expected_k = ref.qk(actual_qkv[1], a + "k_norm", pos, KH)
                check(layer, "k_norm_rope", keys[pos:], expected_k)
                keys = keys.repeat_interleave(QH // KH, 1).transpose(0, 1)
                values = values.repeat_interleave(QH // KH, 1).transpose(0, 1)
                scores = (
                    actual_q.transpose(0, 1).float()
                    @ keys.transpose(1, 2).float()
                    / D**0.5
                )
                low = (
                    torch.arange(pos, pos + R, device="cuda") - c["sliding_window"] + 1
                )
                scores.masked_fill_(
                    torch.arange(pos + R, device="cuda")[None, None, :]
                    < low[None, :, None],
                    -torch.inf,
                )
                expected_attn = (
                    (scores.softmax(-1) @ values.float())
                    .transpose(0, 1)
                    .reshape(R, Q)
                    .to(torch.bfloat16)
                )
                actual_attn = stages(layer, "attention", Q // W, True)
                check(layer, "attention", actual_attn, expected_attn)
                projected = ref.parts(actual_attn, a + "o_proj")
                for rank, (actual, expected) in enumerate(
                    zip(stages(layer, "o_local", H), projected)
                ):
                    check(layer, f"o_local_{rank}", actual, expected)
                actual_o = stages(layer, "o", H)[0]
                check(layer, "o_fold", actual_o, ref.fold(stages(layer, "o_local", H)))
                residual, n = ref.add_norm(
                    residual, actual_o, base + "post_attention_layernorm"
                )
                actual_n = stages(layer, "ffn_norm", H)[0]
                check(layer, "ffn_norm", actual_n, n)
                gate = stages(layer, "gate", I // W, True)
                up = stages(layer, "up", I // W, True)
                check(layer, "gate", gate, ref.linear(actual_n, base + "mlp.gate_proj"))
                check(layer, "up", up, ref.linear(actual_n, base + "mlp.up_proj"))
                codes, scales = ref.mx_swiglu(
                    torch.cat((gate, up), -1), c["swiglu_alpha"], c["swiglu_limit"]
                )
                scales = ref.mx_scales(codes, scales)
                for rank, actual in enumerate(stages(layer, "down_local", H)):
                    width = I // W
                    expected = ref.native_product(
                        codes[:, rank * width : (rank + 1) * width],
                        scales[:, rank * width // 32 : (rank + 1) * width // 32],
                        None,
                        base + "mlp.down_proj",
                        "mxfp8",
                        rank * width,
                        1.0,
                        True,
                    )
                    check(layer, f"down_local_{rank}", actual, expected)
                pending = stages(layer, "down", H)[0]
                check(
                    layer,
                    "down_fold",
                    pending,
                    ref.fold(stages(layer, "down_local", H)),
                )
            _, expected = ref.add_norm(residual, pending, PREFIX + "final_norm")
            actual = load(block / "hidden.bf16", np.uint16, (R, H))
            check(-1, "final_norm", actual, expected)
            reports.append(dict(case=case.name, block=block.name, stages=rows))
            args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
