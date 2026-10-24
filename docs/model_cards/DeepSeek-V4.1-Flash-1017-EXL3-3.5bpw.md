# DeepSeek-V4.1-Flash-1017 EXL3 3.5 bpw on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw),
an EXL3 trellis re-pack of
[DeepSeek-V4.1-Flash-1017](DeepSeek-V4.1-Flash-1017.md) made with
exllamav3 1.6.2 (revision `04a492fc`, 44 shards, 239.9 GiB). The engine
decodes the trellis itself inside its projection kernels: no
dequantization pass, no intermediate checkpoint, the pack as published.
The port's plan, format contract and record are in
[docs/deepseek_v41_flash_exl3_plan.md](../deepseek_v41_flash_exl3_plan.md).

## The checkpoint, as loaded

Every projection of the model — attention (`wq_a`, `wq_b`, `wkv`, the
per-group `wo_a.slice.g`, `wo_b`), the indexer's `wq_b` and `wk`, the
compressors, the routed and shared experts, the Engram projection, the
DSpark drafts' projections and experts, the lm head — is one EXL3 matrix:
a `trellis` of int16 words (`[K/16, N/16, 16 × bits]`, a 16 × 16 tile of
trellis-coded values per entry), an input scale vector `suh` (f16, per
input channel) and an output scale vector `svh` (f16, per output channel),
under the `mul1` codebook (its marker tensor is checked at load). The bit
widths come from the pack's `exl3_k_map.json`, which the config parser
requires beside `config.json`:

| Class | Bits |
|---|---|
| Routed experts, layers 0–9 and 30–39 | 4 |
| Routed experts, layers 10–29 | 3 |
| Shared experts, attention, compressors, indexer `wq_b`, Engram `wkv`, drafts, head | 6 |
| Indexer `wk` | 8 |

The tensors that are not matrices changed dtype in the re-pack: the
routers, `weights_proj`, the Markov head and the confidence head are f16,
`gate.bias` is f16, the Engram gate weights are f32. The loader takes an
f16 or f32 native only when its value is exactly representable in the
engine's bf16 form (it checks every element and refuses otherwise); the
f16 biases and confidence head are widened to f32. The pack ships no
Engram tables: the deployment names the base release through
`engine.engram_source`, and the loader admits only the two table tensors
from that snapshot (its Engram hash sidecar serves too). Everything else —
tokenizer, prompt renderer, DSML tool calls, the cache formats — is
V4.1-Flash-1017's.

The binding census on the published pack (`dsv41_bind_check`, headers
only): 192,495 tensors matched with their dtypes and shapes, nothing
unexpected, and exactly the four Engram table tensors missing — the ones
the base release supplies.

## The numerics

A product with an EXL3 matrix is `svh ∘ H_N (H_K (x ∘ suh) · W_inner)`
with 128-point Hadamard rotations over √128 on both axes and `W_inner`
the codebook values. The engine's chain: the bf16 activation row is scaled
by `suh` and rotated in fp32, scaled, and rounded to fp16 — the one
rounding the format adds — then multiplied by the fp16 codebook values on
the tensor cores with fp32 accumulation in ascending k; the outputs are
rotated and scaled by `svh` in fp32 and rounded once at the site (bf16 or
f32). The row's chain does not depend on how many rows ride in a launch
(1 to 128 rows take one weight stream — the 128-row A tile since
2026-10-23; larger prompts run 128-row groups with the same chain), so
the batch family's row invariance holds here as for the other formats. The numpy reference (`tools/dsv41_reference_dump.py`)
decodes the trellis independently and models the same single fp16
rounding; the forward chain on the EXL3 fixture passes the strict
teacher-forced budgets of the release fixture (every layer's streams
within 0.6 % relative l2, the kept rows' top-1 exact, the DSpark rows
within 1 %).

Tensor parallelism slices every matrix on whole 128-blocks of the rotated
axis: at world 4 the experts' intermediate of 2,304 (18 blocks) splits
640 / 640 / 512 / 512 on even-indexed experts and 512 / 512 / 640 / 640
on odd ones (the shared expert by its layer's parity), so a step's expert
bytes even out across the ranks; the head's 129,280 rows 253 / 253 / 252
/ 252 blocks, the attention's query heads and output groups as the
release's.
World 4 is the deployment that fits: the template's memory plan reads
81.8 GiB on rank 0 at sixteen slots (the resident weights with the
replicated embedding, the 14 GiB prefix arena and the 1M-token pool;
81.5 GiB at twelve slots, 87.5 at six before the expert slices were
balanced across the ranks on 2026-10-23), against ~116 GiB free per node.

## Serving

Four DGX Spark nodes, tensor parallel, the 1017 template's shape
([deploy/cluster_deepseek-v4.1-flash-1017_exl3-3.5bpw_w4.example.json](../../deploy/cluster_deepseek-v4.1-flash-1017_exl3-3.5bpw_w4.example.json)):
sixteen request slots at DSpark depth 5 — the block's five drafts, the
confidence-scheduled verify depth choosing among 1, 3 and 5 of them a
step (depth 4 until the evening of 2026-10-23: the fifth draft pays on
code, tables, JSON and counting, where the head's survivals stay high
through the block) — (the family's decode batch is 96 rows since the evening of
2026-10-23 — 64 and twelve slots that morning, 32 and six slots before:
the aggregate past six streams is where the pack trailed most), the
bounded prefill with a 4,096-token first chunk for a
lone prompt (`engine.prefill_initial_tokens`), the full 1M-token context,
the indexer budget at the checkpoint's 512, and `engine.engram_source:
"deepseek-ai/DeepSeek-V4.1-Flash-1017"` — that repository must be in every
node's model cache (its 48 shards; only shards 47 and 48 are read).

The pack is lossy against the release (3.5 bits per weight on average
against the release's MXFP4 experts and FP8 dense stack); on the
task-level evaluations (`scripts/serve_eval.py`) it is indistinguishable
from the release — the campaign below reads HumanEval 159/164, GSM8K
293/300 (294 in the morning's six-slot run), schema extraction 100/100
against the release's 157 / 295 / 100.

## Measured (2026-10-24, four nodes): the campaign of record

The full row set of [benchmarks](../benchmarks.md) on source `b423d5db` plus the day's working tree, binary
`9a179b15a6de02fa`, the sixteen-slot template with the five-draft DSpark block, in one `serve_campaign.py run`
with the five DSpark mode boots (2026-10-24 00:06–00:54 UTC; every rank at zero swap before and after, zero cached
tokens in the 12 cold probes, matching operation streams in 7/7 launches, the plain boot's transcripts identical
to the default's in 5/5 classes, 9/9 API checks, isolation identical). The middle column is the previous day's
campaign on the twelve-slot template with the expert-run slot form (binary `6f8bccfa5b70efef`, 12:56–13:45 UTC,
the same gates); the last is the release's MXFP4/FP8 template on the same shape (six slots, DSpark depth 4
scheduled, the 1M context), its campaign of 2026-10-20 (`283a52f7`, binary `f4e9e9cdb16acf86`). Two more
campaigns of 2026-10-23 are quoted in the text where they separate the steps: the six-slot template at 05:41 UTC
(binary `c7efb833f88a06e3`: C1 67.0 / 91.5 / 105.0 / 93.1 / 65.0 tok/s by class, C8 91.1–137.9) and the
sixteen-slot template at depth 4 on this same binary at 19:26 UTC (C1 71.0 / 91.6 / 121.5 / 112.1 / 71.5, C8
165.3–242.4).

| | EXL3 3.5bpw, sixteen slots, five drafts (`9a179b15a6de02fa`) | EXL3 3.5bpw, twelve slots, run form (`6f8bccfa5b70efef`) | MXFP4/FP8 release, 2026-10-20 (`283a52f7`) |
|---|---|---|---|
| C1 decode tok/s (prose / code / JSON / math / chat) | 70.7 / 92.0 / 134.2 / 111.4 / 70.1 | 70.6 / 96.2 / 119.3 / 111.3 / 70.9 | 53.5 / 80.0 / 89.4 / 87.5 / 55.0 |
| Decode tok/s by concurrency, C1 / C2 / C4 / C8 | 70.1–134.2 / 95.7–182.4 / 127.8–206.6 / 165.0–247.7 | 70.6–119.3 / 96.7–170.3 / 126.4–183.7 / 167.7–224.0 | 53.5–89.4 / 75.5–121.9 / 94.7–130.3 / 83.8–139.0 |
| Wall tok/s, C1 / C2 / C4 / C8 | 67.6–124.3 / 92.9–170.4 / 124.3–195.6 / 160.3–235.8 | 68.1–111.1 / 93.5–159.5 / 122.8–174.7 / 162.7–214.1 | 52.0–84.6 / 73.4–115.8 / 92.2–125.2 / 81.7–132.7 |
| Cold prefill s, 2K / 8K / 32K | 1.188 / 3.446 / 13.331 | 1.166 / 3.398 / 13.326 | 1.006 / 2.638 / 10.024 |
| Context decode ms per pass (tok/s), short / 32K / 64K / 128K / 256K / 512K | 34.70 (91.2) / 37.80 (95.0) / 38.14 (85.7) / 40.67 (81.4) / 44.39 (74.6) / 52.38 (65.8) | 33.18 (94.3) / 35.59 (93.0) / 37.37 (85.3) / 38.78 (78.3) / 42.81 (77.4) / 50.58 (59.3) | 43.12 (73.0) / 44.82 (72.9) / 45.64 (69.5) / 48.49 (64.9) / 51.99 (53.3) / 60.35 (51.5) |
| Context prefill s, short / 32K / 64K / 128K / 256K / 512K | 0.272 / 12.470 / 25.652 / 54.885 / 125.507 / 345.781 | 0.260 / 12.516 / 25.622 / 54.724 / 125.425 / 345.913 | 0.263 / 9.684 / 19.999 / 43.577 / 103.139 / 301.219 |
| Decode modes, C1 tok/s: plain / DSpark 1 / 2 / 3 / 5 / adaptive | 52.6–53.0 / 65.0–76.6 / 69.2–99.3 / 67.0–110.9 / 58.3–135.1 / 70.1–134.2 (≤5) | 52.9–53.1 / 65.0–75.9 / 71.2–100.6 / 66.8–107.6 / 57.2–129.8 / 70.6–119.3 (≤4) | 41.7–41.9 / 50.2–59.5 / 50.1–76.5 / 47.8–82.6 / 39.6–96.7 / 53.5–89.4 (≤4) |
| HumanEval / GSM8K / extraction | 160/164 / 295/300 / 100/100 | 159/164 / 293/300 / 100/100 | 157/164 / 295/300 / 100/100 |

What the second day's two programs bought between the twelve-slot and the sixteen-slot columns. The decode batch
96 rows and sixteen slots (the multi-row program of 2026-10-23 evening, [plan §3.6](../deepseek_v41_flash_exl3_plan.md)):
the C8 cell is eight live streams on a sixteen-slot world — its step at 40 rows 91 → 82.5 ms on the counting
prompt — and sixteen identical streams decode at 128 ms a step where the twelve-slot template queued four. The
five-draft block under the schedule's three options (1, 3, 5), on the same binary as the depth-4 sixteen-slot
campaign: JSON 121.5 → 134.2 tok/s at C1 (4.25 → 5.10 tokens a step), the C2 / C4 / C8 tops 174.9 → 182.4 /
195.0 → 206.6 / 242.4 → 247.7, prose 71.0 → 70.7, code 91.6 → 92.0, math 112.1 → 111.4 and chat 71.5 → 70.1
(all four within the three-repeat spread; the prose and chat passes are the same 29–31 ms at four or five drafts,
the schedule holding them at one to three rows). Against the twelve-slot column: C1 JSON +12 %, code −4 % (its
transcript drafts 3.19 tokens a step here against 3.04 at depth 4 and 96.2 tok/s on the twelve-slot world's
shorter step), prose, math and chat level; C2 +7 %, C4 +12 % and C8 +11 % at the top.

What the first day's levers bought, cell by cell between the six-slot and the twelve-slot campaigns. The C8 cell took two steps:
twelve slots made it eight live streams instead of six live and two queued (prose 91.1 → 136.8, code 131.3
→ 158.5, JSON 137.9 → 174.8, math 119.3 → 162.2, chat 96.0 → 148.2 tok/s on binary `dde85abfaa953cc1`), and
the expert-run slot form then took it to 167.7 / 200.8 / 224.0 / 208.3 / 177.9 (the step 71.6–128.3 ms for
12.0–28.7 tokens): an expert several rows pick is now streamed and trellis-decoded once a step instead of
once a slot. The same form is the C1 gain — prose 67.0 → 70.6, code 91.5 → 96.2, JSON 105.0 → 119.3, math
93.1 → 111.3, chat 65.0 → 70.9 (the pass 27.8–35.6 ms at 1.96–4.25 tokens; the twelve-slot campaign before
the form read 66.4 / 89.0 / 110.0 / 101.6 / 66.4): the shared expert's five-row run and the draft block's own
duplicate picks. C2 +6–20 % and C4 +9–25 % by class follow. The plain one-row pass 20.0 → 18.9 ms (the deep
slot ring, the balanced slices, the fused reduce's launches gone); the context matrix 2–8 % less a pass from
a short prompt to 512K. Prefill 1.225 / 3.489 / 13.607 → 1.166 / 3.398 / 13.326 s (−5 / −3 / −2 %: the
4,096-token first chunk at 2K, the 128-row dense tile at 8K and 32K), the 512K prompt 350.7 → 345.9.

Against the release: the plain pass 18.9 ms against 23.9 (+27 % tok/s), and the drafted cells follow it —
C1 +20 % (code) to +33 % (JSON) by class, C2 +28–40 %, C4 +33–41 %, the context matrix 16–23 % less a pass
from a short prompt to 512K; the C8 cell 167.7–224.0 against 83.8–139.0 is twelve slots against six (the
release's C8 is six live and two queued, with that cell's three-repeat spread measured in the release's card,
"The six-slot template at C6 and C8").

Quality: 159/164, 293/300, 100/100 against the release's 157/164, 295/300, 100/100 and the morning's 159 /
294 / 100 — within the sets' repeat noise (the release's own two campaigns read 157 / 295 and 157 / 294). The
3.5 bpw trellis costs nothing these three tasks measure. Six responses hit the 2,048-token cap (five
HumanEval, one GSM8K); at the 8,192 retry the five HumanEval items completed and passed, the GSM8K item
completed and failed.

Prefill is the column the release keeps: 1.166 / 3.398 / 13.326 s against 1.006 / 2.638 / 10.024 (+16 / +29 /
+33 %), the 512K prompt 345.9 s against 301.2 (+15 %); per token ~1,730 / 2,340 / 2,400 tok/s at 2K / 8K /
32K. Of the plan's second prefill round ([§3.4](../deepseek_v41_flash_exl3_plan.md)) the 128-row A tile
landed this day; the deeper B ring and the reconstruct-to-fp16 tensor-core form for the large dense chunks
remain open, with the short-prompt floor (0.52 ms a token at 3K against 0.42 at 47K: the per-walk costs of
the folds, the Engram gather, the row copies and the attention over a short ring).

The decode modes: fixed depth 5 wins JSON and math (135.1 / 118.0 against the schedule's 134.2 / 111.4) and
loses code, prose and chat (88.3 / 58.3 / 65.8 against 92.0 / 70.7 / 70.1); the schedule takes 70.1–134.2
without a per-class choice. Its constants are the pack's own pass costs since 2026-10-23 evening
(`engine.mtp_schedule_base_ms` 19.9, `engine.mtp_schedule_row_ms` 2.95, from the mode boots' curve) with three
depth options (`engine.mtp_schedule_options` 3: 1, 3 and 5 of the five drafts).

Before the morning's run, the first requests on the pack exposed a decode-graph defect, fixed before any row
was taken (the plan's §7 has the record): `GlmMoeLayer::prepare_graph_table` wrote the routed EXL3 expert
views into the graph's per-slot table but had shared-expert branches for the NVFP4 and packed-int forms only,
so every graph step ran each MoE layer's shared expert on a stale view — the first token after a prefill
right, the text degrading within a few tokens, the eager engine right. The MoE test's EXL3 decode case now
runs the graph table (bitwise the host chain) and the engine test's exact graph-vs-eager comparisons stand.

## License and attribution

The EXL3 format, its `mul1` codebook and the decoding procedure follow
exllamav3 (MIT, Turboderp; [docs/licenses/exllamav3.md](../licenses/exllamav3.md)).
The engine's kernels, loader and reference are DGPP's own implementations
of that contract. The weights are the re-pack's authors' redistribution of
DeepSeek's release under the terms stated in the pack's repository.
