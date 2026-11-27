# DeepSeek-V4.1-Flash on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
It is **not** a republished checkpoint: the engine loads the upstream
release as shipped, with no re-quantization pass. The quantization study
(`docs/deepseek_v41_flash_plan.md` §1.11) found nothing to gain over the
released encoding, so the three community NVFP4 re-packs are not used.

## The checkpoint, as loaded

475.2 GiB across 48 shards. The routed and draft experts are MXFP4 (e2m1
nibbles with an e8m0 scale per 32 along K); the dense and attention
projections are FP8 e4m3 with e8m0 scales on 32×32 blocks; the two Engram
n-gram tables are FP8 e4m3 with an e8m0 scale per 32 (94.56 GiB each,
mapped read-only from the NVMe rather than resident); the embedding and
LM head are BF16. The engine decodes every code exactly into BF16. Decode
rows run BF16 activations. Prompt rows follow `engine.prefill_activations`:
BF16, or with `native` the release's own recipe, which rounds the input of
every FP8 and MXFP4 multiply to FP8 with a power-of-two scale per 32
values. The KV caches keep the trained fp4/fp8 formats. There is no
`kv_dtype` knob for this family.

## Architecture

40 layers as a 20-layer causal encoder and a 20-layer decoder (the CED
split). CSA2 attention gives every layer a 128-token sliding window and,
from the compressor layers on, a compressed-KV path whose positions a
two-level indexer selects (512 of them per query). The residual stream is
four hyper-connection copies collapsed per sublayer by a single-pass mHC.
Two Engram layers (1 and 14) add n-gram-hash lookups gated into the
stream. The DSpark draft appends three stages that predict a block of five
tokens per decode pass.

## Serving

Four DGX Spark nodes hold the weights at 72.94 GiB per rank (tensor
parallel, world 4); a 128K-token context at two request slots fits in
76.55 GiB with the engine's 4 GiB headroom. The default prefill is
**bounded** (`engine.prefill`): the encoder runs over the whole prompt,
the decoder over its last window — the model's own SWA-replay recipe, about
half the prefill work — with an **exact** 40-layer mode available for
parity. Decode runs a CUDA-graph step with the DSpark block draft
(`mtp_depth` 5). The tokenizer is a three-stage byte-level pre-tokenizer;
the prompt renderer follows the checkpoint's own `encoding/encoding.py`
(there is no Jinja template); tool calls are emitted and constrained in the
model's DSML tag format.

## Measured (2026-09-14, four nodes)

Single stream 76 ms/pass at 2.33 tokens/pass (32.5 ms/token); DSpark
acceptance p1 55–92 % by prompt class; bounded prefill 1.6–2.4 ms/token
over 512–8,192 tokens; gsm8k 60/60, HumanEval 40/40, schema extraction
30/30 (thinking off). The MTP transcript is byte-identical to the plain
greedy transcript, and the four ranks' operation streams match. Full
numbers: `docs/measurements.md` and the 2026-09-14-dsv41-flash measurements.

## License and attribution

The model, its weights, tokenizer and prompt encoder are DeepSeek's; see
the upstream model card for its license and terms. This file describes
only the engine's support and adds no weights.
