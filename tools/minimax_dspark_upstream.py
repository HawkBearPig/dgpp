#!/usr/bin/env python3
"""Replay captured target taps through the actual pinned SGLang DSpark model.

This is a matched-prefix greedy acceptance diagnostic, not a serving benchmark.
The SM121 adapter uses FlashInfer B12x MXFP8 GEMMs and noncausal FA2 attention.
Default context mode rebuilds FC, hidden norm, K/V, K norm, and RoPE from raw
captured target taps, including generated tokens; it never consumes DGPP K/V.
The target trajectory remains DGPP's. --emulate-tp=4 is an additional arithmetic
control for BF16 local outputs and FP32 reduction, not a distributed SGLang run.
"""

REVISION = "58484246d26523eaf1a7f16ab524deee288d2120"
SOURCE_SHA256 = "d61e56fa29d173d0726cfce0d48145cdef8cc5ad4dabdace299a4e111fa352eb"


def phase(ids, think_start, think_end):
    for token in reversed(ids):
        if token == think_start:
            return "thinking"
        if token == think_end:
            return "answer"
    return "unmarked"


def prefix_length(predictions, truth):
    for index, (actual, expected) in enumerate(zip(predictions, truth)):
        if actual != expected:
            return index
    return min(len(predictions), len(truth))


def summarize(results):
    report = {}
    for case in ["all"] + sorted({r["case"] for r in results}):
        rows = [r for r in results if case == "all" or r["case"] == case]
        report[case] = {}
        for label in ["all"] + sorted({p for r in rows for p in r["phases"]}):
            report[case][label] = {}
            for model in ("proposals", "upstream"):
                positions = []
                for k in range(1, 8):
                    eligible = [
                        r
                        for r in rows
                        if len(r["truth"]) >= k
                        and (label == "all" or r["phases"][k - 1] == label)
                    ]
                    reached = sum(
                        r[model][: k - 1] == r["truth"][: k - 1] for r in eligible
                    )
                    accepted = sum(r[model][:k] == r["truth"][:k] for r in eligible)
                    positions.append(
                        dict(
                            position=k,
                            eligible=len(eligible),
                            reached=reached,
                            accepted=accepted,
                            conditional_pct=100 * accepted / reached
                            if reached
                            else None,
                        )
                    )
                report[case][label][model] = positions
    return report


def main():
    import argparse
    import hashlib
    import importlib.metadata
    import json
    import sys
    import tempfile
    import types
    from pathlib import Path
    import numpy as np
    import torch
    from functools import partial

    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument(
        "--sglang-source",
        type=Path,
        required=True,
        help="Pinned checkout root, containing python/sglang",
    )
    ap.add_argument("--capture-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gemm-backend", default="b12x")
    ap.add_argument("--attention-backend", default="fa2")
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--context", choices=("captured", "upstream"), default="upstream")
    ap.add_argument("--emulate-tp", type=int, choices=(1, 4), default=1)
    a = ap.parse_args()
    if a.output.exists():
        raise ValueError("Choose a new output file; existing reports are never reused")
    if a.blocks < 1:
        raise ValueError("--blocks must be positive")
    source = a.sglang_source / "python/sglang"
    source_hash = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        source_hash.update(
            str(path.relative_to(source)).encode() + b"\0" + path.read_bytes()
        )
    if source_hash.hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the unmodified pinned SGLang Python source tree")
    receipt = json.loads(a.capture_receipt.read_text())
    assert (
        receipt["complete"]
        and receipt["returncode"] == 0
        and len(receipt["ranks"]) == 4
    )
    for rank in receipt["ranks"].values():
        for probe in ("first", "last"):
            assert (
                rank[probe]["swap_kib"] == 0
                and rank[probe]["sha256"] == receipt["binary_sha256"]
            )
    assert (
        a.engine_dir.resolve()
        == Path(
            receipt["command"][receipt["command"].index("--dump-dir") + 1]
        ).resolve()
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(complete=False, results=[])) + "\n")
    # Skip SGLang's unrelated public frontend import; all SRT/model/kernel modules
    # below come from the complete, hash-checked upstream source tree.
    pkg = types.ModuleType("sglang")
    pkg.__path__ = [str(source)]
    sys.modules["sglang"] = pkg
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.configs.minimax_vl import (
        MiniMaxM3VLConfig,
        configure_minimax_m3_dspark,
    )
    from sglang.srt.models.dspark_minimax import DSparkMiniMaxDraftModel
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )
    from sglang.srt.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
        ParallelLMHead,
    )
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
        MarkovGreedyStep,
    )
    from minimax_reference_dump import Weights
    import flashinfer

    checkpoint = a.checkpoint_dir
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    think_start = tokenizer.token_to_id("<mm:think>")
    think_end = tokenizer.token_to_id("</mm:think>")
    assert think_start is not None and think_end is not None
    engine = a.engine_dir
    publish(ServerArgs(model_path=str(checkpoint / "dspark")), role="test")
    rendezvous = tempfile.TemporaryDirectory(prefix="dgpp-dspark-upstream-")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="file://" + rendezvous.name + "/group",
        backend="nccl",
    )
    initialize_model_parallel()
    parallel = get_parallel().override(
        tp_rank=0,
        tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_rank=0,
        attn_cp_size=1,
    )
    parallel.__enter__()
    config = configure_minimax_m3_dspark(
        MiniMaxM3VLConfig(**json.loads((checkpoint / "dspark/config.json").read_text()))
    )
    quant = Fp8Config.from_config(config.quantization_config)
    torch.set_default_dtype(torch.bfloat16)
    w = Weights(checkpoint / "dspark")
    target = Weights(checkpoint)

    def tensor(weights, name):
        value = torch.from_numpy(np.array(weights.raw(name), copy=True))
        dtype = weights.header[name]["dtype"]
        return (
            value.view(torch.bfloat16 if dtype == "BF16" else torch.float8_e4m3fn)
            if dtype in ("BF16", "F8_E4M3")
            else value
        )

    with torch.device("cuda"):
        model = DSparkMiniMaxDraftModel(config, quant).eval()
    print("UPSTREAM MODEL CONSTRUCTED", flush=True)
    with torch.no_grad():
        model.load_weights((name, tensor(w, name)) for name in w.header)
        for name, module in model.named_modules():
            qm = getattr(module, "quant_method", None)
            if qm is not None:
                qm.process_weights_after_loading(module)
                if isinstance(qm, Fp8LinearMethod) and qm.use_mxfp8:
                    # SM121 adapter: execute FlashInfer's official B12x backend with the
                    # same published MXFP8 operands/layout, avoiding SM100-only GEMM dispatch.
                    qm.w8a8_mxfp8_linear = partial(
                        flashinfer_mxfp8_blockscaled_linear, backend=a.gemm_backend
                    )
        print("UPSTREAM WEIGHTS LOADED", flush=True)
        with torch.device("cuda"):
            embed = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, params_dtype=torch.bfloat16
            )
            head = ParallelLMHead(
                config.vocab_size, config.hidden_size, params_dtype=torch.bfloat16
            )
        embed.weight.copy_(tensor(target, "language_model.model.embed_tokens.weight"))
        head.weight.copy_(tensor(target, "language_model.lm_head.weight"))
        model.attach_shared_modules(embed_tokens=embed, lm_head=head)
        print("TARGET EMBEDDING AND HEAD ATTACHED", flush=True)

    # Arithmetic control only: run row-parallel GEMMs independently, then reproduce
    # DGPP's BF16-local / FP32-fold boundary on one GPU. All local GEMMs and fused
    # SwiGLU producers still execute SGLang/FlashInfer code, not DGPP kernels.
    if a.emulate_tp > 1:
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        def parts(base):
            weight = tensor(w, base + ".weight").cuda()
            scales = tensor(w, base + ".weight_scale_inv").cuda()
            width = weight.shape[1] // a.emulate_tp
            return [
                (
                    weight[:, i * width : (i + 1) * width].contiguous(),
                    flashinfer.block_scale_interleave(
                        scales[:, i * width // 32 : (i + 1) * width // 32].contiguous()
                    ),
                )
                for i in range(a.emulate_tp)
            ]

        def product(x, weight, scales):
            if isinstance(x, Mxfp8SwizzledInput):
                return flashinfer_mxfp8_blockscaled_linear(
                    x.data, weight, scales, input_scale=x.scales, backend=a.gemm_backend
                )
            return flashinfer_mxfp8_blockscaled_linear(
                x, weight, scales, backend=a.gemm_backend
            )

        def fold(values):
            output = values[0].float()
            for value in values[1:]:
                output = output + value.float()
            return output.to(torch.bfloat16)

        def row_forward(shards):
            def forward(x):
                return fold(
                    [
                        product(part.contiguous(), *weights)
                        for part, weights in zip(x.chunk(a.emulate_tp, -1), shards)
                    ]
                ), None

            return forward

        def mlp_forward(mlp, shards):
            def forward(x, forward_batch=None):
                gate_up, _ = mlp.gate_up_proj(x)
                gate, up = gate_up.chunk(2, -1)
                values = []
                for g, u, weights in zip(
                    gate.chunk(a.emulate_tp, -1), up.chunk(a.emulate_tp, -1), shards
                ):
                    fused = mlp._swiglu_oai_mxfp8(torch.cat((g, u), -1))
                    assert fused is not None, "native fused SwiGLU required"
                    values.append(product(fused, *weights))
                return fold(values)

            return forward

        for i, layer in enumerate(model.layers):
            prefix = f"language_model.model.dspark.layers.{i}.decoder_layer."
            layer.self_attn.o_proj.forward = row_forward(
                parts(prefix + "self_attn.o_proj")
            )
            mlp = layer.decoder_layer.mlp
            mlp.forward = mlp_forward(mlp, parts(prefix + "mlp.down_proj"))
        print("TP4 ROW-PARALLEL ARITHMETIC CONTROL ENABLED", flush=True)

    def read(path, dtype, shape):
        x = torch.from_numpy(
            np.fromfile(path, dtype=dtype).reshape(shape).copy()
        ).cuda()
        return x.view(torch.bfloat16) if dtype == np.uint16 else x

    class OfflineAttention:
        def __init__(self, case, block, pos, context=None):
            if context is not None:
                self.context = context
                return
            self.context = []
            for layer in range(config.num_hidden_layers):
                self.context.append(
                    tuple(
                        torch.cat(
                            [
                                read(
                                    case / f"rank{r}" / block / f"{name}-{layer}.bf16",
                                    np.uint16,
                                    (pos + 7, 1, 128),
                                )
                                for r in range(4)
                            ],
                            1,
                        )[:pos].contiguous()
                        for name in ("k", "v")
                    )
                )

        def forward(self, q, k, v, layer, fb, save_kv_cache=True, **kwargs):
            ck, cv = self.context[layer.layer_id]
            keys = torch.cat((ck, k))
            values = torch.cat((cv, v))
            return flashinfer.single_prefill_with_kv_cache(
                q.reshape(-1, 64, 128),
                keys,
                values,
                causal=False,
                sm_scale=layer.scaling,
                backend=a.attention_backend,
            ).reshape(len(q), -1)

    def project_context(path, rows, start):
        taps = [
            read(
                path / f"target-layer-{layer}.bf16",
                np.uint16,
                (rows, config.hidden_size),
            )
            for layer in config.dflash_config["target_layer_ids"]
        ]
        normalized = model.project_target_hidden(torch.cat(taps, -1))
        positions = torch.arange(start, start + rows, device="cuda", dtype=torch.int64)
        result = []
        for layer in model.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(normalized)
            k = attn.apply_k_rope(positions, attn.apply_k_norm(k))
            result.append((k.reshape(rows, 4, 128), v.reshape(rows, 4, 128)))
        return result

    results = []
    try:
        with torch.no_grad():
            for case in sorted(engine.iterdir()):
                if not (case / "rank0").is_dir():
                    continue
                transcript = np.fromfile(
                    case / "rank0/generated.i32", dtype=np.int32
                ).tolist()
                prompt_ids = np.fromfile(
                    case / "rank0/prompt.i64", dtype=np.int64
                ).tolist()
                prompt = len(prompt_ids)
                context = (
                    project_context(case / "rank0", prompt, 0)
                    if a.context == "upstream"
                    else None
                )
                blocks = sorted(
                    (case / "rank0").glob("block-*"),
                    key=lambda p: int(p.name.split("-")[1]),
                )
                if len(blocks) < a.blocks:
                    raise ValueError(
                        f"{case.name}: fewer than {a.blocks} captured blocks"
                    )
                for block in blocks[: a.blocks]:
                    meta = json.loads((block / "metadata.json").read_text())
                    pos = meta["position"]
                    rows = 7
                    ids = torch.tensor(
                        [meta["anchor"]] + [config.mask_token_id] * 6,
                        device="cuda",
                        dtype=torch.int64,
                    )
                    assert (
                        rows == config.block_size == 7
                        and meta["native"]
                        and meta["world"] == 4
                    )
                    assert (
                        config.num_attention_heads == 64
                        and config.num_key_value_heads == 4
                        and config.head_dim == 128
                    )
                    assert pos + rows <= config.sliding_window, (
                        "Offline adapter currently supports contexts inside the full attention window"
                    )
                    assert transcript[pos - prompt] == meta["anchor"]
                    positions = torch.arange(
                        pos, pos + rows, device="cuda", dtype=torch.int64
                    )
                    fb = ForwardBatch(
                        forward_mode=ForwardMode.EXTEND,
                        batch_size=1,
                        input_ids=ids,
                        req_pool_indices=torch.tensor(
                            [0], device="cuda", dtype=torch.int64
                        ),
                        seq_lens=torch.tensor(
                            [pos + rows], device="cuda", dtype=torch.int64
                        ),
                        out_cache_loc=positions,
                        seq_lens_sum=pos + rows,
                        positions=positions,
                        extend_num_tokens=rows,
                        extend_seq_lens=torch.tensor(
                            [rows], device="cuda", dtype=torch.int32
                        ),
                        extend_prefix_lens=torch.tensor(
                            [pos], device="cuda", dtype=torch.int32
                        ),
                        extend_start_loc=torch.tensor(
                            [0], device="cuda", dtype=torch.int32
                        ),
                        extend_seq_lens_cpu=[rows],
                        extend_prefix_lens_cpu=[pos],
                    )
                    if context is not None:
                        assert all(len(k) == pos for k, v in context)
                    backend = OfflineAttention(case, block.name, pos, context)
                    with forward_context(ForwardContext(backend), forward_batch=fb):
                        output = model(ids, positions, fb)
                        logits, _ = model.compute_base_logits(output.hidden_states)
                    previous = meta["anchor"]
                    predictions = []
                    for row in logits:
                        prev = model.markov_head.get_prev_embeddings(
                            torch.tensor([previous], device="cuda")
                        )
                        previous = int(
                            MarkovGreedyStep.execute(
                                base_logits=row[None],
                                prev_embeds=prev,
                                w2_weight=model.markov_head.markov_w2.weight,
                            )[0]
                        )
                        predictions.append(previous)
                    old = np.fromfile(block / "proposals.i32", dtype=np.int32).tolist()
                    truth = transcript[pos - prompt + 1 : pos - prompt + 8]
                    assert old[: meta["accepted"]] == truth[: meta["accepted"]]
                    if meta["accepted"] < 7:
                        assert old[meta["accepted"]] != truth[meta["accepted"]]
                    prefix = prompt_ids + transcript[: pos - prompt + 1]
                    phases = [
                        phase(prefix + truth[:i], think_start, think_end)
                        for i in range(len(truth))
                    ]
                    hidden = read(block / "hidden.bf16", np.uint16, (7, 6144))
                    rms = float(
                        (
                            (hidden.float() - output.hidden_states.float())
                            .square()
                            .mean()
                            / hidden.float().square().mean()
                        ).sqrt()
                    )
                    record = dict(
                        case=case.name,
                        block=block.name,
                        position=pos,
                        proposals=old,
                        upstream=predictions,
                        truth=truth,
                        phases=phases,
                        hidden_relative_rms=rms,
                    )
                    results.append(record)
                    print(
                        case.name,
                        block.name,
                        "accepted",
                        prefix_length(old, truth),
                        prefix_length(predictions, truth),
                        flush=True,
                    )
                    a.output.write_text(
                        json.dumps(
                            dict(
                                sglang_revision=REVISION,
                                reference_script_sha256=hashlib.sha256(
                                    Path(__file__).read_bytes()
                                ).hexdigest(),
                                sglang_python_sha256=SOURCE_SHA256,
                                dgpp_source_revision=receipt["source_revision"],
                                dgpp_binary_sha256=receipt["binary_sha256"],
                                versions={
                                    name: importlib.metadata.version(name)
                                    for name in (
                                        "torch",
                                        "flashinfer-python",
                                        "sglang-kernel",
                                    )
                                },
                                device=torch.cuda.get_device_name(),
                                checkpoint=str(checkpoint.resolve()),
                                summary=summarize(results),
                                complete=False,
                                gemm_backend=a.gemm_backend,
                                attention_backend=a.attention_backend,
                                upstream_tp=1,
                                emulated_row_parallel_width=a.emulate_tp,
                                dgpp_tp=4,
                                context=a.context,
                                results=results,
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    if context is not None:
                        added = project_context(block, 8, pos)
                        keep = meta["accepted"] + 1
                        context = [
                            (torch.cat((ck, k[:keep])), torch.cat((cv, v[:keep])))
                            for (ck, cv), (k, v) in zip(context, added)
                        ]
    finally:
        parallel.__exit__(None, None, None)
        torch.distributed.destroy_process_group()
        rendezvous.cleanup()
    report = json.loads(a.output.read_text())
    if not report["results"]:
        raise ValueError("No captured cases found")
    report["complete"] = True
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
