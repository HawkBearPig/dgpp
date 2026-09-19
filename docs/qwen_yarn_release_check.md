# The YaRN 512K release check

Date: 2026-09-18

`scripts/qwen_yarn_release_check.py` is the opt-in, real-checkpoint acceptance
run for `engine.rope_scaling` (the Qwen3.8-Flash-Next YaRN recipe,
[qwen38_flash_next_plan.md](qwen38_flash_next_plan.md) §1.9.1). It is a **release
check, not a test**: it never runs in CI, it needs an idle two-node cluster and
about an hour, and it refuses to start without `--run` (or
`DGPP_YARN_RELEASE_CHECK=1`). This page is the procedure: what has to be true
before it, how to run it, and how to read what it prints.

## Why it exists

The gates already in the tree prove the arithmetic:
`tests/unit/qwen_rope_scaling_test.cpp` freezes the inverse frequencies, the
cos/sin scale and the correction band against vLLM's YaRN, `tests/cuda/qsa_test.cu`
checks the tables the kernels build, and the plan check in
`tests/cuda/qwen_forward_test.cpp` shows the memory plan is the same number with
the knob on and off. None of those says that the engine, on a real checkpoint,
*serves* a 524 288-token request: that the long prefill finishes, that decode
keeps its pace, that the prefix cache is reused across requests that share the
document, that four streams at once still answer, and that the model still finds
a needle it was told at 5 % or 95 % of a 512K context. That is a measurement on
hardware, and it belongs next to the deployment, not in a CI job.

## Prerequisites

- **The 2× Spark kit.** Two GB10 nodes, `.env` site settings (`DGPP_NODES`,
  `DGPP_SSH_USER`, the ports), RoCE selected and quiet
  ([networking.md](networking.md)), `nvidia/Qwen3.8-Flash-Next-NVFP4` complete
  in each node's local cache (`scripts/dgpp-cluster doctor` proves it). Run
  everything below on the head node — that is rank 0, and it is where
  `nvidia-smi` gives the peak-memory column.
- **A build with serving enabled** (`dgpp_serve_app` → `build-release/dgpp-serve`),
  at the commit that carries the knob.
- **The deployment**: copy
  [`../deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json`](../deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json)
  without `.example` and resolve it as usual. It is the two-Spark NVFP4 shape
  with `engine.rope_scaling` on, two request slots and a 532 480-token pool. The
  KV-pool and headroom notes are in
  [`../deploy/README.md`](../deploy/README.md).
- **Optionally, the reference lane**: the vLLM container the q-dgx-gateway
  recipe starts as **lane Q** for this checkpoint — the same
  `nvidia/Qwen3.8-Flash-Next-NVFP4`, launched with `YARN_ENABLE=true
  YARN_FACTOR=2.0 MAX_MODEL_LEN=524288`. Lane Q carries the YaRN settings from
  the launcher's `--hf-overrides`, so it is the same recipe by construction; if
  you change one lane's factor you change both. Bring it up on its own port and
  pass `--reference-url`. Running both lanes at once on the same two nodes is
  possible (each is ~55 GiB per rank) but the timings stop meaning anything, so
  run them one after the other, or run the reference lane on the kit's second
  pair.

## Running it

```bash
CONFIG=deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.json
cp deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json "$CONFIG"

# 1. The plan first: it must fit under the 4 GiB headroom on both ranks.
build-release/dgpp-serve --config "$CONFIG" --memory-plan   # or via dgpp-cluster up

# 2. The world up, on an otherwise idle cluster.
python3 scripts/dgpp-cluster up --config "$CONFIG"

# 3. The check. --run is the acknowledgement; without it nothing happens.
python3 scripts/qwen_yarn_release_check.py --run \
    --lengths 262144 524288 --concurrency 4 \
    --json-out /tmp/yarn512k-$(git rev-parse --short HEAD).json
    # add: --reference-url http://LANE_Q_HOST:PORT

# 4. The world down.
python3 scripts/dgpp-cluster down --config "$CONFIG"
```

Before it sends a request the check reads `/v1/models` on both endpoints and
refuses to continue when the DGPP lane's advertised request limit is below the
longest document it is about to send (a `262144` there means the knob did not
reach the process, and the run would only measure truncation), or when the two
lanes advertise different limits. `--allow-ceiling-mismatch` overrides both for
the case where you *want* that comparison.

The default seed (`20260918`) fixes every prompt: the filler records, where the
needles sit and what their codes are. Two runs at the same seed ask the same
questions of the same positions, which is what makes a diff between a DGPP build
and lane Q, or between this build and last week's, readable.

## What one length does

For each of `--lengths` (262 144 and 524 288 by default), on each lane:

| phase | what it is | what it records |
| --- | --- | --- |
| `retrieval` | one deterministic document of the target size, then one greedy question per needle depth (0.05 … 0.95). The first probe is the **cold prefill**: the engine's own `prefill_ms` and computed-token counters, and the client's TTFT. | `hits/requests`, `hit_rate`, `cold_prefill_ms`, `cold_prefill_ms_per_token`, per-probe `prompt_tokens`, `cached_tokens`, `answer` |
| `cache_reuse` | the first probe again, byte for byte, so the shared prefix has to come from the arena | `cached_ratio`, the second `prefill_ms` |
| `decode` | an incremental decode: one long generation on an 8 192-token document | `ttft_ms`, `ms_per_token`, `completion_tokens`, `finish` |
| `concurrent` | four streams over the same document at once | per-stream `ttft_ms`, the median, `aggregate_tokens_per_s`, `errors` |
| `short_context` | once per lane, the same probes at `--control-length` (4 096) | the same retrieval columns |

Peak memory is sampled from `nvidia-smi` on the node the script runs on, every
`--mem-sample-s` (2 s), and the record names that node. Take the other rank's
number with `scripts/node_probe.sh` while the long prefills run and paste it in
beside the plan's total; the check does not guess it.

## Reading the results

The JSON's `verdict.checks` is the pass/fail summary; the printed block repeats
it. Exit status 0 is a pass, 1 a failed criterion, 2 the missing `--run`.

With a reference lane, retrieval is attributed, not judged:

| observation | reading |
| --- | --- |
| both lanes miss the same needle | the model or the YaRN recipe, not a DGPP regression — record the depth and move on |
| lane Q hits, DGPP misses | a **DGPP regression**: the tables, the mscale, the correction band or the cache path. Re-run the unit gates, then diff the probe's `prompt_sha256` against the reference's (identical means the same bytes went in) |
| DGPP hits, lane Q misses | put it in the record; it is a reason to re-check lane Q's recipe, not to ship silently |
| short-context misses on DGPP only | the more serious shape of the same bug: something scaled that should not have. The 4 096-token control has no long-context excuse |

The timing columns are reported, not gated — a release check that guesses its own
latency thresholds has no business printing PASS. Compare them against the
previous record for the same kit: cold `ms_per_token` should track the plain
262 144 run's (the ramp changes the rope table and nothing else), `decode
ms_per_token` should be the build's ordinary T=1/T=2 pace, and `cache_reuse
cached_ratio` near zero means `engine.prefix_cache_gib` is 0 on one of the two
lanes.

For the plan, quote the memory-plan total next to the peak: the 512K shape is
51.73 GiB per rank with 2 slots at `kv_capacity` 532 480, plus the 4 GiB the
pre-flight check adds, and the YaRN knob itself moves nothing
(§1.9.1: the same plan with the ramp on and off is the same number).

## What it does not do

It does not run in CI, does not install anything, and does not bring the world up
or down — read the boot log for the two YaRN lines (`serve: request context
limit …`, and the refusal if the family is not `qwen4_exp`) yourself. It does not
test a checkpoint that already declares YaRN in its own `rope_parameters`; that
shape is rejected at load today
([qwen38_flash_next_plan.md](qwen38_flash_next_plan.md) §1.9.1). It does not
sweep `factor`, `beta_fast`/`beta_slow`, `attn_factor` or `mrope_cache_factor`:
one recipe, two lengths, one verdict.
