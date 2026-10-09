# Swift 1.5 Qwen3.8-Flash-Next NVFP4 on two Sparks (2026-10-09)

**Status: community conversion, validated on one two-Spark deployment; no engine
changes.**

`kaushikvira/Qwen3.8-Flash-Next-swift15-nvfp4-dgpp` is a ModelOpt-format NVFP4
conversion of [ukisai/Swift1.5-Qwen3.8-Flash-Next](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)
(the reasoning-efficiency fine-tune of Qwen3.8-Flash-Next), produced for the
two-Spark deployment. It is the **same format and layout** as
`nvidia/Qwen3.8-Flash-Next-NVFP4`, so it is served by the existing NVFP4 path with
one deployment template
([`cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.example.json`](../../deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.example.json)).

Being a conversion rather than a code path, this record is a **reference for the
configuration**, not a claim about the fine-tune's quality. All model credit to
ukisai (Swift 1.5) and the Qwen team (base).

## Artifact

| | |
|---|---|
| Repository | `kaushikvira/Qwen3.8-Flash-Next-swift15-nvfp4-dgpp` (public) |
| Size | 125.87 GiB of tensors (128 GB on disk), 52 shards, 296,475 tensors |
| Expert tensors | 294,912 — NVFP4 e2m1 codes (2/byte) + e4m3 block-16 scales, encoded with `src/loaders/nvfp4_quant.hpp` (bit-identical to the loader-side encoder) |
| n-gram table | `model-plefp8-00000.safetensors`: 128 F8_E4M3 shards + BF16 `weight_scale` (the source ships BF16 with no scale; re-encoded at `ws = absmax/448 = 0.000199317932`) |
| Dense / norms | copied byte-for-byte (1,560 tensors); served with `dense_weights: "fp8"` |
| MTP draft experts | the source's BF16 fused pair, `mtp_expert_format: "bf16_fused"` (the engine encodes them to FP8 at load) |
| Hashes | `SHA256SUMS` (64 files) |

The layout transform is the only non-trivial step: the source stores the MoE
experts fused and stacked (`layers.N.mlp.experts.gate_up_proj [512, 2I, H]`,
`.down_proj [512, H, I]`, 98 tensors); the engine reads them per-expert. The
split (`gate = rows [0:I]`, `up = rows [I:2I]`) was verified against the nvidia
NVFP4 release: cosine 0.9967 for the correct pairing, 0.0086 crossed.

## Configuration

The template above: two Sparks, BF16 K/V, `ngram_table: "mmap"`,
`dense_weights: "fp8"`, `fp8_head: "mma"`, MTP depth 4 with
`mtp_expert_format: "bf16_fused"`, 1,048,576-token pool, 42 GiB prefix cache,
4,096-token busy/idle prefill budgets, four request slots.

Bring-up: `qwen_load_check` validates the whole binding (`opened in 2.4 s`,
49 layers, 69.20 GiB resident at world 1, n-gram table mmap 47.68 GiB at scale
`0.00019931793`); the serve API gate passes in full; the checkpoint's 296,475
tensors carry no duplicates and cover every name the `nvidia/Qwen3.8-Flash-Next-NVFP4`
reference has, except its per-expert FP8 MTP tensors, which the BF16-fused pair
replaces.

## Throughput (wall tokens/s, median of three)

`serve_load.py --concurrency 1,2,4 --max-tokens 256 --repeat 3 --classes all`,
against the `nvidia/Qwen3.8-Flash-Next-NVFP4` release on the same pair, same
config, in the same session.

| class | C1 nvidia | C1 swift15 | C2 nvidia | C2 swift15 | C4 nvidia | C4 swift15 |
|---|---:|---:|---:|---:|---:|---:|
| prose | 54.9 | 59.7 | 83.2 | 79.1 | 116.6 | 114.4 |
| code | 86.0 | 92.3 | 132.3 | 126.9 | 174.2 | 164.6 |
| json | 97.6 | 100.6 | 157.4 | 156.3 | 201.1 | 192.4 |
| math | 82.5 | 81.9 | 135.4 | 137.3 | 159.9 | 169.2 |
| chat | 53.9 | 52.9 | 82.8 | 86.2 | 126.2 | 123.8 |

The deltas are inside the lane's session-to-session spread (the nvidia lane's own
recorded baseline for the prose row is C1 54.4 / C2 83.1 / C4 116.4). Same
architecture and the same NVFP4 recipe — no throughput difference is expected.

## llama-benchy 0.4.0 (engine-native OpenAI API, 3 runs, `--exact-tg`)

pp 2048/8192 × tg 256/512 × context depth 0 / 16k / 32k, greedy, each lane booted
in turn. Each row is one llama-benchy entry (a pp phase and a tg phase).

| test | base pp | swift15 pp | Δpp | base tg | swift15 tg | Δtg | base ttfr | swift15 ttfr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pp2048/tg256@0 | 1979.7 | 1932.7 | −2.4% | 61.0 | 59.8 | −2.1% | 1037 ms | 1062 ms |
| pp2048/tg512@0 | 1962.3 | 1925.9 | −1.9% | 61.7 | 60.2 | −2.5% | 1046 ms | 1066 ms |
| pp8192/tg256@0 | 1856.4 | 2255.4 | +21.5% | 55.7 | 59.5 | +6.8% | 5133 ms | 3634 ms |
| pp8192/tg512@0 | 1944.7 | 2294.0 | +18.0% | 61.0 | 60.4 | −1.0% | 4256 ms | 3573 ms |
| pp2048/tg256@16k | 2030.4 | 2255.4 | +11.1% | 59.1 | 61.4 | +4.0% | 9086 ms | 8176 ms |
| pp2048/tg512@16k | 2122.3 | 2273.2 | +7.1% | 60.5 | 57.2 | −5.5% | 8712 ms | 8110 ms |
| pp8192/tg256@16k | 2212.7 | 2265.0 | +2.4% | 58.5 | 56.2 | −4.0% | 11138 ms | 10854 ms |
| pp8192/tg512@16k | 2286.4 | 2280.4 | −0.3% | 61.2 | 59.9 | −2.2% | 10752 ms | 10779 ms |
| pp2048/tg256@32k | 2261.0 | 2245.2 | −0.7% | 55.6 | 53.8 | −3.4% | 15401 ms | 15509 ms |
| pp2048/tg512@32k | 2233.8 | 2230.1 | −0.2% | 61.6 | 56.5 | −8.4% | 15594 ms | 15616 ms |
| pp8192/tg256@32k | 2245.7 | 2204.0 | −1.9% | 58.5 | 56.8 | −2.9% | 18242 ms | 18602 ms |
| pp8192/tg512@32k | 2244.4 | 2238.5 | −0.3% | 72.6 | 59.7 | −17.8% | 18252 ms | 18300 ms |

Prefill stays at ~1.86–2.29k tok/s and decode at ~54–73 tok/s in both lanes. The
large positive pp deltas at shallow depth and the negative tg deltas go in both
directions across the matrix and are prefix-cache warmup / MTP-acceptance
variance. The engine's own record at depth 0 on this pair: MTP 2.9 tok/step,
acceptance p1 73% / p2 52% / p3 36% / p4 28%.

## Token efficiency (not reproduced)

The upstream model card claims **63.4% fewer thinking tokens / 1.8×**. A
five-prompt `reasoning_effort: xhigh` probe (cap 24,576, one seed) on this pair:

| prompt | nvidia tokens | swift15 tokens | Δ |
|---|---:|---:|---:|
| rancube (random walk, exact answer) | 1,510 | 782 | −48.2% |
| counting | 2,107 | 2,083 | −1.1% |
| physics (rolling cylinder) | 2,436 | 2,436 | 0.0% |
| modpow | 24,576 (cap) | 24,576 (cap) | 0.0% |
| logic (zebra puzzle) | 17,675 | 24,576 (cap) | +39.0% |
| total | 48,304 | 54,453 | +12.7% |

One large win, three neutral, and one where the fine-tune ran to the cap. **The
card's 31–57% reductions were not reproduced**; its protocol is a standardized
benchmark suite (GPQA-Diamond, MMLU-Pro, AIME, HMMT, ERQA, LiveCodeBench) with
five seeds at temperature 1.0 / top_p 0.95 / top_k 20, which was not run here.
Two of the five prompts were also truncated by the cap. Read the deployment as
throughput-equivalent to the nvidia release; treat the token-efficiency claim as
unconfirmed on this hardware.

A greedy transcript comparison (`serve_greedy_transcript.py --max-tokens 4096`,
temperature 0) shows −24.1% completion tokens and −24.6% wall on the chat prompt,
−2.4% on json, −3.5% on math, and both models hitting the cap on code.

## Reproduce

```bash
cp deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.example.json \
   cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.json
python3 scripts/dgpp-cluster.py up --config cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.json

# throughput
python3 scripts/serve_load.py 127.0.0.1 18080 --concurrency 1,2,4 \
  --max-tokens 256 --repeat 3 --classes all

# llama-benchy (engine-native OpenAI API)
llama-benchy --base-url http://127.0.0.1:18080/v1 --api-key none \
  --model Qwen3.8-Flash-Next --tokenizer <this checkpoint> \
  --pp 2048 8192 --tg 256 512 --depth 0 16384 32768 --runs 3 --exact-tg
```

## Caveats

- **Licensing**: Swift 1.5 is `swift-open-license-1.0` and its source snapshot is
  gated; this artifact is a redistribution of a derivative. Check ukisai's terms.
- The template points at a community repository under one account — if it is
  removed, the template breaks.
- `input_scale` is `1.0` for every expert matrix: correct for W4A16 (this path
  never reads the activation scale), not calibrated for W4A4.
- The MTP draft experts are BF16 on disk (the engine encodes them at load), so
  the artifact is 2.35 GiB larger than it would be with the nvidia release's
  per-expert FP8 MTP tensors.
- No quality/correctness campaign was run against this conversion (no GSM8K /
  HumanEval / IFBench), unlike the curated records in this directory.
