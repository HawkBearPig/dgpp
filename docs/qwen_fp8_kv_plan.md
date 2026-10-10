# Qwen3.8-Flash-Next — FP8 KV cache (plan)

Branch: `feat/qwen-fp8-kv-cache` (off `master`). Scope: **Qwen3.8-Flash-Next only**
(`src/models/qwen/` + the QSA kernels + the Qwen serve family). GLM / MiMo / DSV4 are
out of scope — they already have their own `kv_dtype` paths. **`model35.cpp` is also
out of scope:** `QwenFullAttnLayer` (`full_attn_decode` / `full_attn_prefill`,
`src/kernels/full_attn.cu:482,513`, built only in `model35.cpp:665`) reads a
same-shaped `k_cache`/`v_cache` but rides a *different* pool struct
(`layers.hpp:325`) and is Qwen3.5-only — do not "fix" it as part of this work.

**Status (2026-10-10): Phase A is implemented and verified.** Commits on
`feat/qwen-fp8-kv-cache`: `95e08f66` (fp8 pool + config wiring), `a57d99d0`
(append quantize-on-write), `b96ac756` (in-kernel fp8 dequant for all three
attention kernels + the real-path test). The bf16 path is bitwise-unchanged
(`qsa_test` 12/12); `engine.kv_dtype: fp8` now boots the Qwen lane.

## 0. Goal

Add an `engine.kv_dtype` knob to the Qwen lane so the QSA paged K/V cache can be
stored in **FP8 E4M3**, with a native, dequant-free tensor-core attention path as
the end state.

**The win, in real numbers** (measured from the live plan line,
`log/deployments/c1c3cdef3b168073/serve_r1.log`: "kv cache pool … **13.81 GiB**"
per rank, 13 QSA layers × 1 kv-head/rank × dim 256, `kpool=4`, `idx_dim=128`,
`kv_capacity=1048576`, `max_concurrency=4`):

| | bf16 (today) | fp8 |
|---|---|---|
| K/V per layer | 1.000 GiB | 0.500 GiB |
| scales per layer (2 × `slots·kv_heads·4 B`) | — | 0.008 GiB |
| index per layer (unchanged, bf16) | 0.0625 GiB | 0.0625 GiB |
| **per layer** | **1.0625 GiB** | **0.5703 GiB** |
| **13 layers, per rank** | **13.81 GiB** | **7.41 GiB** |

- K/V planes alone: **1.97×**; the whole pool (index + rings stay bf16): **1.86×**.
- Frees **6.40 GiB per rank / 12.8 GiB across both Sparks** → headroom 8.4 →
  **~14.8 GiB free**.
- Crisp framing: *fp8 at 2 000 000 pool tokens costs what bf16 at 1 048 576 costs
  today* → with 4 requests capped at the native 256K context, that is **8 slots at
  256K instead of 4** (no YaRN needed), or push `prefix_cache_gib` 42 → 48. The pool
  is per-rank, so the win is counted twice on a TP=2 lane.

### Why FP8 E4M3 (the "native, no-dequant" answer)

The GB10 is Blackwell (sm_121a). Its 5th-gen tensor cores natively consume
**E4M3** (and E2M1/FP4) operands directly in a matmul. So the QSA attention can
feed the stored E4M3 K/V codes straight into the FP8 tensor cores for `Q·Kᵀ` and
`P·V`, applying the per-row fp32 scales in the epilogue — **no bf16 dequant step**.

- **FP8 E4M3** = target. 1 B/element (half of bf16's 2). The de-facto standard
  KV-cache format (vLLM / SGLang / TRT-LLM all do FP8 KV). Native to the cores.
- **FP4 E2M1 (NVFP4)** = also native and half the bytes again, but a much bigger
  quality hit for K/V and non-standard. Stretch goal only, not this plan.

### Two honesty constraints

1. The engine's existing **MiMo fp8 KV is storage-only**: it stores E4M3 + a
   per-(slot, kv-head) fp32 scale and **dequants to bf16 inside the attention
   kernel** before the matmul (`kernels/mimo_attn.cu` `decode_fp8_piece`). That is
   the cheap, safe first step — *not* the native no-dequant path.
2. **Phase A is sharper than "within tolerance."** MiMo's fp8 kernel over a cache
   is **bitwise** the bf16 kernel over the *dequantized* rows
   (`docs/mimo_v26_flash_plan.md:223`; `mimo_attn_test` pins both). Phase A
   inherits that: quantize-on-write + dequant-in-read means the only divergence
   source is the quantizer, not kernel reordering. So the **kernel-level** test
   can be an exact-match-against-dequantized-bf16 test; keep the tolerance band
   only for the *transcript* gate.

## 0b. Numerical evidence (pre-implementation)

Built **before** any fp8 kernel, because Phase A's divergence is *purely the
quantizer* (the isolation result above) — so it can be measured by round-tripping
the existing bf16 K/V through the e4m3 codec and pushing it through the existing
bf16 attention. No new kernel is needed to establish the band.

### The bounds (analytic)

- **Per-element (e4m3, RNE, 3-bit mantissa):** relative quantization error
  **≤ 2⁻⁴ (6.25 %) worst case, ~2⁻⁵ expected** — for *representable* values. The
  per-row scale `s = absmax/448` is the tightest that keeps the row's max at
  e4m3's 448 ceiling (no overflow, max utilization). Caveat: values far below the
  row's `absmax` underflow to 0 (relative error → 1), but their magnitude is a
  negligible fraction of `absmax`, so they contribute nothing to the L2 norm.
- **Score `s = q·k/16`** (256-dim dot, k quantized): worst case
  `|Δs| ≤ 2⁻⁴·|s|`; the *expected* error grows as `√256` (independent RNE
  errors), not `256` → the typical score error is ~8× smaller than the worst case.
- **Softmax:** 1-Lipschitz in logit differences → a score error `Δs` perturbs a
  probability by `≈ e^{Δs} − 1 ≈ Δs`.
- **`P·V`** (v quantized, `Σpᵢ = 1`): `|Δo| ≤ 2⁻⁴·Σpᵢ|vᵢ| ≤ 2⁻⁴·‖v‖` → the
  **attention-output relative error is O(2⁻⁴) worst case**, typically far smaller.

**Scope of the bound:** this bounds the *attention layer* (a **necessary
condition**). It does **not** bound the 48-layer compounding or the **argmax**
(a **sufficient condition**) — those are validated empirically by the gate +
bench, not proven. Nobody should read "within tolerance" as a theorem about the
final token.

### The evidence harness — `tests/cuda/qwen_fp8_kv_attn_test.cu`

Two tests, run via `ctest -R qwen_fp8_kv_attn` (GPU):

1. **`qwen_fp8_kv_per_element_codec_study`** (host): round-trips 4096 realistic
   bf16 K/V rows (unit-normal) through `latent_fp8_row_scale` / `latent_fp8_encode`
   / `latent_fp8_decode_bf16`; reports the rel-error distribution (p50/p90/p99/max)
   and the underflow fraction. The 3-bit-mantissa RNE bound (2⁻⁴) holds for
   *representable* values; the rare sub-scale values underflow to 0 (rel error → 1)
   but are a negligible fraction of the row's `absmax`, so the honest assertion is
   **p99 ≤ 2⁻⁴** plus "underflowed `|x|/absmax` ≤ 0.01".
2. **`qwen_fp8_kv_attention_sample_match`** (GPU): builds a paged QSA cache from
   bf16 K/V and a second from the e4m3-round-tripped K/V; runs the **existing**
   `qsa_attn_partial` + `dsa_attn_combine` + `qsa_gate_out` over both; reports
   `l2_rel`, `max_abs`, and the **sample-match rate** (fraction of output
   elements within 2⁻⁴ / 2⁻³ relative) and asserts `l2_rel ≤ 2⁻⁴` (the
   necessary-condition bound).

**Use of the numbers:** the sample-match test's reported band is the
**pre-registered acceptance threshold** for Phase A — implement the fp8 storage
path, then check it against *these* numbers (not "see if the gate passes").

### Measured (2026-10-10, GB10, `build-release`)

```
[fp8-kv] per-element e4m3 rel error (n=1048576): p50 0.0216  p90 0.0422  p99 0.0553  max 1.0000;
         underflow 0.001% of elems (|x|/absmax <= 2.1e-06)
[fp8-kv] attention sample-match (n=24576): l2_rel 0.0368  max_abs 0.02344
         match(<=2^-4) 67.22%  match(<=2^-3) 82.93%
```

Reading: p99 (0.0553) is under the 2⁻⁴ mantissa bound; the only "max 1.0" is the
0.001 % of sub-scale values that underflow to 0 (|x|/absmax ≤ 2.1e-6, negligible).
The attention output's `l2_rel` is **0.0368 (3.7 %)** — well under the 2⁻⁴
necessary-condition bound, confirming the √N averaging in the score. **Pre-
registered Phase A band: attention `l2_rel` ≤ 0.0625; expect ~0.03–0.04.**

### Measured on the REAL path (2026-10-10, after Phase A)

The `qwen_fp8_kv_real_path_matches_bf16` test exercises the actual fp8 pool
(`qsa_kv_append` quantizes on write; the attention kernels dequant in-kernel)
against the bf16 pool, for all three attention kernels:

```
[fp8-kv] REAL PATH (append-quantize + in-kernel dequant) vs bf16: l2_rel 0.0360  max_abs 0.02441
[fp8-kv] REAL PATH prefill kernel vs bf16:                        l2_rel 0.0360  max_abs 0.02441
[fp8-kv] REAL PATH warp prefill kernel vs bf16:                   l2_rel 0.0360  max_abs 0.02974
```

All three within the pre-registered band, and matching the host-round-trip
prediction (0.0368) — the quantizer is the only divergence source, as designed.

## 1. Phases

- **Phase A — storage-only FP8 (dequant-to-bf16 in attention).** Store K/V as
  E4M3 + per-(slot, kv-head) fp32 scale; dequant in the QSA attention read path.
  Mirrors the proven MiMo pattern + reuses `kernels/latent_format.hpp` codecs.
  Low risk, **1.86× the pool** (−6.40 GiB/rank), kernel-bitwise-vs-dequantized-
  bf16.

## 2. Storage layout

Per QSA layer, per rank (from `QwenKvPoolShape`: `kv_heads`, `dim=256`):

| plane | bf16 (today) | fp8 (new) |
|---|---|---|
| K | `uint16 [slots, kv_heads·dim]` | `uint8 [slots, kv_heads·dim]` (E4M3) |
| V | `uint16 [slots, kv_heads·dim]` | `uint8 [slots, kv_heads·dim]` (E4M3) |
| K scale | — | `float [slots, kv_heads]` (absmax/448 per head-row) |
| V scale | — | `float [slots, kv_heads]` |
| index_cache (compressed keys) | bf16 | **keep bf16** (small: `idx_dim=128`, one per 4 tokens; scorer stays bf16) |
| ring (pending raw keys) | bf16 | **keep bf16** (decode pool assembly reads it) |

Rationale for keeping index/ring bf16: the big win is K/V (`dim=256`, 2 kv
heads); the indexer keys are 128-wide and one per 4 tokens, and the block
scorer is a dot product that's simplest to leave bf16. Revisit only if the pool
math demands it.

Use the existing `kernels/latent_format.hpp` `LatentFormat::kFp8` codecs
(`latent_fp8_row_scale`, `latent_fp8_encode`, `latent_fp8_decode_bf16`) — the
same per-row `absmax/448` recipe MiMo uses, so the codes are consistent.

## 3. File-by-file changes

### 3.1 Config knob (already half-done) + a pre-work guard
- `src/serve/cluster_config.cpp` — `engine.kv_dtype` is **already** parsed (line
  182). **No change to parsing.**
- `src/serve/cluster_config.hpp` — `e.kv_dtype` string already exists. No change.
- **Validator caveat (E4):** validation is `latent_format_from_string`, which
  also accepts the aliases `fp8_e4m3`, `e4m3`, `nvfp4`, `e2m1`, `fp8_block`,
  `fp4_block` (`latent_format.hpp:66-72`). Today `kv_dtype: "nvfp4"` or
  `"fp8_block"` parse clean and are **silently ignored** by the Qwen lane
  (`kv_format_name()` reports bf16, the pool still costs 13.81 GiB/rank) — a user
  can believe fp8 KV is active when it is not. **Pre-work (do first,
  independent of the rest):** have `QwenFamily` throw on any non-`kBf16` value,
  mirroring `MimoFamily` (`dgpp_serve.cpp:834`):
  ```cpp
  if (kv_format != dgpp::LatentFormat::kBf16)   // until Phase A lands
    throw std::invalid_argument(...);
  ```

### 3.2 Serve family — thread the format into the model
`apps/dgpp_serve.cpp` — the `QwenFamily` (≈ line 422–528) today hardcodes
`kv_format_name()` → `"bf16"` at **line 472 only** (line 540 is `Glm4Family`,
out of scope — do not touch it) and builds `QwenModel` without a format.
Mirror the `MimoFamily` pattern (lines 826–866):
- add `dgpp::LatentFormat kv_format` member; validate `kBf16 | kFp8` (reject
  `kFp4` — not implemented);
- pass `kv_format` into the `QwenModel` ctor (line 497) and into
  `QwenModel::plan_memory` (line 485);
- `kv_format_name()` → `dgpp::latent_format_name(kv_format)`;
- **`make_family` call site (E3, would not compile without it):**
  `make_family` already takes `kv_format` (`dgpp_serve.cpp:963`) and forwards it
  to MiMo/GLM, but the Qwen branch at **line 972** drops it:
  `return std::make_unique<QwenFamily>(ckpt, rope_scaling, fp8_head_mma);`
  → add `kv_format` to the `QwenFamily` ctor and to this call.
- Read the value from the parsed engine settings (`e.kv_dtype` →
  `latent_format_from_string`).

### 3.3 Model ctor + pool shape
- `src/models/qwen/forward.hpp` (line 129) / `forward.cpp` (line 52): add
  `LatentFormat kv_format = LatentFormat::kBf16` to the `QwenModel` ctor and to
  `plan_memory`; store it; feed it into the `QwenKvPoolShape` at both build sites
  (forward.cpp ≈ 182 and ≈ 337).
- `src/models/qwen/kv_pool.hpp` — `QwenKvPoolShape`: add
  `LatentFormat format = LatentFormat::kBf16`.

### 3.4 Pool — allocate the fp8 planes
`src/models/qwen/kv_pool.hpp` / `kv_pool.cpp`:
- `QwenKvPoolShape` gains `format`.
- `QwenKvPool`: when `format == kFp8`, allocate K/V as `uint8` (1 B/elem) plus
  `float` scale planes `[slots, kv_heads]`; keep index/ring bf16. Add
  `k_scale_` / `v_scale_` members.
- `cache_bytes()`: fp8 → `kv = token_slots·kv_heads·dim·1` (1 B/elem) **plus two
  scale planes, each `token_slots·kv_heads·4 B`** (8 B per slot-head total);
  bf16 unchanged. This is what the memory plan reports.
- `view()`: populate the new `QwenQsaCache` scale pointers (null for bf16).
- `reset_all()` / `reset_request()`: zero the scale planes too.
- `copy_block_contents()`: copy the scale planes alongside K/V (prefix-cache
  block copy must move the scales with the codes).
- `validate()` (a **file-local free function** in an anonymous namespace,
  `kv_pool.cpp:11`, not a member): reject `kFp4` (not implemented).

### 3.5 Cache view struct
`src/models/qwen/layers.hpp` — `QwenQsaCache` (line 234): add
`float* k_scale = nullptr; float* v_scale = nullptr;` (null ⇒ bf16 path).

### 3.6 KV append — quantize on write
`src/kernels/qsa.cu` `qsa_kv_append` (line 770) + `qsa.hpp`:
- When scales are non-null, quantize each written K/V
  head-row to E4M3 with `latent_fp8_row_scale`/`latent_fp8_encode` and store the
  per-head scale (mirror `mimo_attn.cu` `store_fp8_head`, lines 166/180).
- bf16 path (scales null) is byte-for-byte unchanged.

### 3.7 Attention read path — the phase split
The serving path (`src/models/qwen/layers.cpp:803,807`) calls **`qsa_attn_partial`**
(`qsa.hpp:132`) for decode and short prefill, and **`qsa_attn_prefill_warp`**
(`qsa.hpp:160`) for long prefill. `qsa_attn_partial` is a thin wrapper that
forwards to `qsa_attn_partial_gather` (`qsa.hpp:143`, `qsa.cu:875-881`,
`async=-1`) — the change lands in **one body** either way; cite both entry
points.

**Key fact (G1): decode and short-prefill have NO tensor cores today.** They are
warp-shuffle FMA dot products (`qsa.cu:274-279,627`, `qsa_prefill.cu:109` —
`__shfl_xor_sync` reductions). Only the long-prefill path uses MMA:
`qsa_warp.cu:55` `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with
`ldmatrix` (`:46-51`).

- **Phase A (dequant-to-bf16):** where each kernel loads a K/V row, dequant
  E4M3 → bf16 with `latent_fp8_decode_bf16(code, scale)` (mirror
  `mimo_attn.cu` `decode_fp8_piece`, lines 324/337) *before the existing FMA / MMA
  math* (there is no "bf16 matmul" in the decode/short-prefill kernels — they are
  shuffle-FMA). The downstream math is untouched.

### 3.8 Memory plan label
`src/models/qwen/forward.cpp` (line 346): the `plan.add("kv cache pool (K/V
bf16, …)")` string + `cache_bytes` now reflect the format (fp8 → smaller).

## 4. Verification (per phase)

Gate order: unit → parity → memory → bench → A/B.

1. **Unit (codecs + pool):** extend `tests/cuda/qwen_*` — a fp8 round-trip test
   (quantize a K/V row, dequant, assert within `2^-4` rel of bf16, using
   `latent_quantize_row_host` / `latent_dequantize_row_host` as the oracle).
   Pool test: `cache_bytes` for fp8 == half of bf16 for K/V (+ the two scale
   planes); scale planes sized `[slots, kv_heads]`; `copy_block_contents` moves
   scales.
2. **Parity (the real gate):** run via **`make gate` in `~/work/dgpp-gateway`**
   (`gates/run-gates.sh` → `scripts/serve_api_check.py` +
   `scripts/serve_greedy_transcript.py` + `scripts/serve_prefill_probe.py`; there
   is **no** `gates/` dir in the dgpp repo).
   - Phase A: the **kernel-level** test is an **exact match against the
     dequantized-bf16 reference** (MiMo precedent, §0.3) — quantize-on-write +
     dequant-in-read, so divergence is only the quantizer. The **transcript**
     gate uses a tolerance band (per-element `~2^-4`; expect the same tokens for
     the gate prompts; document any divergence).
3. **Memory plan:** `make up` on the fp8 lane must show the KV pool line drop
   from **13.81 → ~7.41 GiB/rank** (1.86×, not a clean 2×) and total plan within
   the headroom (the whole point: fit a bigger `kv_capacity` or more slots).
   Add a **boot-plan assertion** that the pool line actually shrank, so a
   regression to the bf16 allocation is caught at boot.
4. **Bench + A/B:** `make bench` / `make ab` on the fp8 lane vs the bf16 lane.
   Expect: neutral-to-faster decode (fewer KV bytes streamed), same TTFT, and a
   larger usable `kv_capacity`. Record to `results/`.
5. **Regression:** the bf16 lane (default) must be **bitwise unchanged** — the
   fp8 code paths are gated on non-null scales, so a bf16 run touches none of
   them. Verify with an existing bf16 A/B record as the baseline.

## 5. Risks / gotchas

- **Numerics:** FP8 KV is a quality trade, not free. The dequant keeps the
  matmul in bf16 (only storage is fp8) → smallest quality hit; validate on the
  agentic + prose benches, not just the gate.
- **Indexer keys stay bf16** (deliberate). If a later pool
  analysis shows the index plane is worth fp8 too, that's a separate change.
- **Prefix-cache block copy** must move the scale planes with the codes
  (`copy_block_contents`) or shared blocks dequant to garbage.
- **TP=2 fold:** K/V are per-rank slices; the scale planes shard exactly like the
  K/V rows (`local_kv_heads`). No new collective — the boundary reducer is
  untouched.
- **FP4** is rejected by `validate()` for now; the config already accepts the
  string but the Qwen family throws on `kFp4` (like MiMo does on anything but
  bf16/fp8).
- **Graph capture:** every new kernel launch must be deterministic/capturable
  (fixed grids, no host reads) to stay in the decode graph — same contract the
  existing QSA kernels honor.

## 6. Definition of done

Phase A is **functionally complete** (2026-10-10). Status per item:

- [x] `engine.kv_dtype: "fp8"` boots the Qwen lane (verified: pool **7.41 GiB**);
      `make gate` passes (API ALL OK, 4/4 prompts). **Caveat:** the greedy
      transcript **diverges** from bf16 (semantically equivalent paraphrases) —
      the lossy-KV argmax flip; not token-identical (see §7.3).
- [x] bf16 path bitwise-unchanged **at the kernel level** (`qsa_test` 12/12; the
      fp8 code is gated on non-null scales, and the fused K kernel is bitwise
      the chain it replaces). The **end-to-end same-binary A/B** is done: the
      `bf16kv` lane booted on this binary reads prose C1 53.3 vs fp8 53.2 —
      neutral (§7.1).
- [x] Memory plan 13.81 → **7.41 GiB/rank** (verified in the boot log).
- [ ] `make ab` record in `results/` — **not captured** (the run was interrupted
      after serve_load + bench). Re-run when a full record is wanted.
- [x] Docs: the `README.md` row and `docs/operations.md` updated.

**Performance (C1, same session):** prose 50.2 (serial gather) → 50.7
(vectorized) → **53.2** (zero-copy K + single-read append), against the bf16 base
**53.3** — i.e. fp8 is now neutral at C1 with the 1.86x pool. See §7.2.

## 7. Open items (2026-10-10, post-Phase-A)

Phase A is merged and fp8 is the gateway's base lane. Remaining: the `make ab`
record file and the long-context measurement (both evidence, not code).

### 7.1 Measure a CLEAN same-binary A/B — DONE (2026-10-10); fp8 is neutral at C1

The confounded number (prose C1/C2/C4 `50.2/78.8/110.9` vs the bf16 base
`53.3/82.9/126.4`, an older session/binary) is superseded. The `bf16kv` lane
was booted with *this* binary (`g44dde933`, the zero-copy build `7b2f1dee`)
and ran the same `serve_load`: prose C1 **bf16 53.3 vs fp8 53.2** — neutral.
The ~5–12 % "penalty" was the confound plus the two now-fixed penalties (§7.2),
not the fp8 storage format. The fp8 record is `results/fp8kv-opt-load.json`
(gateway); the bf16 boot is `log/deployments/ca3552ecc60b4567`.

### 7.2 Remove the fp8 penalties — DONE (2026-10-10); fp8 is neutral at C1

The vectorized gather (`18573bdf`) moved prose 50.2 → 50.7 (+0.9 %, noise), so
byte-throughput was not the bottleneck. The real penalties were the **append**
and the **K double-staging**; commit `7b2f1dee` removed both and recovered the
whole gap:

| fp8 build | prose C1 (tok/s) |
|---|---|
| serial gather | 50.2 |
| vectorized gather | 50.7 |
| **+ zero-copy (fused K, single-read append)** | **53.2** |
| bf16 base | 53.3 |

**1. FIXED — `kv_append_kernel` fp8 quantize: only `kv_heads` threads active.**
Rewritten cooperatively: the block stages the token's K/V rows in smem **once**
(the old per-head form read every row *twice* — an absmax pass then an encode
pass), one warp per head reduces the absmax, all threads encode. `k == nullptr`
skips the K half. Runs on every append.

**2. FIXED — the K went through two bf16 buffers.**
`gemm → k_ → qsa_norm_rope_bf16 → kn_ → append → cache` held a full bf16 K
round trip per token per layer. `qsa_norm_rope_append_fp8` fuses norm + RoPE +
quantize + paged scatter into one kernel writing the cache directly, so `kn_` is
gone (the fp8 QSA path no longer allocates or touches it). **Bitwise the
chain** it replaces (`qwen_fp8_kv_fused_norm_rope_append_matches_chain`).

Still open (minor / long-context):

- **The attention fp8 gather is not pipelined.** The bf16 path issues cp.async
  groups for tile *t+1* under tile *t*'s scores/PV (4 wait points); the fp8 path
  serializes `resolve → load+dequant → scores → pv`. At short context this is not
  measurable (C1 is already neutral), so it is only worth doing if a long-context
  run shows it.
- **Redundant scale reads:** `k_scale[phys*kv_heads+kvh]` is re-read per 16-code
  chunk (16×/row). Hoist to one register per row. Minor.

### 7.3 The expectation "fp8 faster than bf16" is context-dependent

fp8 KV is a **memory** optimization. At short context the K/V is **< 0.1 %** of
the bytes a decode step streams (weights dominate ~6.2 GB/token per rank), so
fp8 can at best be **neutral** on prose C1/C2 — never faster. It becomes a
*speed* win only at long context, where the K/V is a real fraction of the
read traffic (at 128K it is a meaningful slice). So: target **neutral at short
context** (removing 7.2's penalties), and expect the win at long context.
