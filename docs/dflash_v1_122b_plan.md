# DFlash v1 drafter for Qwen3.5-122B-A10B (2026-10-07)

Status: eager and captured proposals are implemented for scalar and
multiple-request execution, including tensor parallelism. Shipped-template
performance campaigns remain pending.

References: `z-lab/Qwen3.5-122B-A10B-DFlash` (snap
`bce6f76c`, 69 tensors, 1.5 GB BF16), the DFlash2 family
(`src/models/qwen/dflash2.*`, `docs/performance_improvement_plan.md`
§7), `docs/adding_a_model.md`.

## Architecture

The checkpoint declares `DFlashDraftModel` (v1): 6 plain Qwen3 layers,
fused `fc` from 8 target taps, shared head — **no selector, no conv,
no codebooks**. The checkpoint architecture selects a separate v1 loader and block forward. The
proposal loop is mask-predict (anchor + 15 masks, per-mask top-1 off the
shared head), not a selector walk.

## Checkpoint layout

`fc.weight` BF16 `[3072, 24576]` (8 taps × 3072, column-split per tap),
`hidden_norm`/`norm` `[3072]`, 6 layers × 11: `input/post_layernorm`
`[3072]`, `self_attn.{q,k,v,o}_proj` (`q` `[4096,3072]` = 32×128, no
`[q|gate]` doubling; `k/v` `[1024,3072]` = 8×128; `o` `[3072,4096]`),
`q/k_norm` `[128]`, `mlp.{gate,up}_proj` `[9216,3072]` +
`down_proj` `[3072,9216]`. All BF16, dense. Config: block 16 (15
drafts), mask id 248077, taps `[1,7,14,20,26,32,39,45]`, hidden 3072,
heads 32/8 × 128, sliding window 4096 (5 sliding + 1 full), rope θ 1e7
default, eps 1e-6, vocab 248320. TP: 32/8 heads and inter 9216 divide
1/2/4; fc taps and norms replicate.

## Execution contract

* New `DFlashConfig` (v1) beside `DFlash2Config` — no silent v1/v2
  merge: arch string, no conv/selector fields, block 16, 8 taps.
* Verify budget: `kSpecRows = 8` caps a step at 7 drafts. The drafter
  proposes 15; the engine verifies the first `dflash_depth` (≤ 7)
  without touching the row budget. Raising the verify cap requires separate acceptance and graph-capacity validation.
* Draft KV rides the pool's draft planes (existing protocol); taps read
  the target's layer outputs; head is the target's `lm_head`.
* The draft loads native BF16 weights; FP8 draft conversion is refused.

## Regression coverage

`qwen35_tp_test` checks request-local draft K/V with different cotenants,
shorter block configurations, the full 16-row reservation at a 64-token
page boundary, warm capture of every batch family, and eager/captured
proposal equality across tensor-parallel ranks. The small reduction buffer
forces chunked projection folds and bounded vocabulary gathers.
`dflash2_kernels_test` checks context-limit padding against an attention
reference that owns only the valid keys.

The recorded draft uses the same stable reduction buffer for each bounded
projection chunk, then copies the reduced rows into the block scratch. The
head's rank-local top-K lists merge with vocabulary offsets before top-1
selection. Every request uses its own block table. Inactive batch slots
stage only padding positions. Admission and reserve account for all query
rows, even when the verify depth is smaller.

The checkpoint provides no acceptance-confidence head, so scheduled verify
depth is refused. Draft proposals are point masses; sampled target
verification retains the existing exact correction rule. BF16 is the only
supported v1 draft-weight format.
