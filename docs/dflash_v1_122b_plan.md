# DFlash v1 drafter for Qwen3.5-122B-A10B (2026-10-07)

Status: stage 1 (config/binding/loader, CPU-only). Proposal loop, verify
wiring and campaign are stages 2–3.

References: `z-lab/Qwen3.5-122B-A10B-DFlash` (snap
`bce6f76c`, 69 tensors, 1.5 GB BF16), the DFlash2 family
(`src/models/qwen/dflash2.*`, `docs/performance_improvement_plan.md`
§7), `docs/adding_a_model.md`.

## 0. Why not DFlash2

The checkpoint declares `DFlashDraftModel` (v1): 6 plain Qwen3 layers,
fused `fc` from 8 target taps, shared head — **no selector, no conv,
no codebooks**. The engine's DFlash path is DFlash2-only (selector walk
+ 2-tap conv + anchor layout) and the parser rejects v1 by design. The
proposal loop is mask-predict (anchor + 15 masks, per-mask top-1 off the
shared head), not a selector walk.

## 1. Census (safetensors header)

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

## 2. Decisions

* New `DFlashConfig` (v1) beside `DFlash2Config` — no silent v1/v2
  merge: arch string, no conv/selector fields, block 16, 8 taps.
* Verify budget: `kSpecRows = 8` caps a step at 7 drafts. The drafter
  proposes 15; the engine verifies the first `dflash_depth` (≤ 7)
  without touching the row budget. Raising the cap (17 rows, variant
  pressure against the 64-limit) is stage 3, only with acceptance data
  past position 7.
* Draft KV rides the pool's draft planes (existing protocol); taps read
  the target's layer outputs; head is the target's `lm_head`.
* Loss: none in stage 1 (native BF16 load). Any `dflash_weights`-style
  recipe is a keyed decision with transcript cost, not a default.

## 3. Stages

* Stage 1 (this commit): plan, `DFlashConfig` parse + `validate_against`
  (MoE target), binding table (69 tensors), BF16 TP loader, unit gates
  incl. a live-header bind check shape test.
* Stage 2: block forward (fused-fc features, 6× sliding bidirectional
  attn + MLP, norms), mask proposal loop (anchor + 15 masks →
  per-mask top-1), `dflash_depth`-capped verify wiring, eager greedy
  C1 + transcript parity vs MTP-depth-2 world.
## 4. Depth ladder (2026-10-07, 416-token prose probe, eager C1)

| depth | tok/pass | accept | ms/pass | tok/s |
|---|---|---|---|---|
| 7 | 2.80 | p1 77 … p7 1 | 152 | 18.5 |
| 4 | 2.69 | p1 79 … p4 15 | 133 | 20.3 |
| 3 | 2.63 | p1 81 p2 52 p3 30 | 108 | 24.4 |
| 2 | 2.28 | p1 80 p2 48 | 90 | 25.2 |
| 1 | 1.80 | p1 80 | 83 | 21.7 |

Depth 2 is the knee (fewer verify rows beat fuller acceptance).
Transcripts identical at every depth. MTP depth 2 stays ahead
(31.1 tok/s, graphed) — v1's gap is launch overhead (stage 3b:
graph capture), not acceptance.
