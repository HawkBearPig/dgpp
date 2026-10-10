# Review — `docs/qwen_fp8_kv_plan.md` (FP8 KV cache for Qwen3.8-Flash-Next)

Date: **2026-10-10**. Reviewed against `~/work/dgpp` @ **`41e3216d`**.
State at review: branch `feat/qwen-fp8-kv-cache` is **byte-identical to `master`**
(`git diff --stat master...feat/qwen-fp8-kv-cache` empty) and the plan doc is
**untracked** — no implementation work has started.

**Verdict: the plan is substantially accurate and still implementable.** The
storage layout, the phase split, the config/serve/pool plumbing and the codec
choices all check out against the tree. It carries **4 factual errors** (2 of
which would block or misdirect the work) and **3 gaps** worth closing before
someone implements it.

Line references below are `file:line` at `41e3216d`; plan references are
`plan:Lnn` into `docs/qwen_fp8_kv_plan.md`.

---

## 1. What checks out

| Plan claim | Verified evidence |
|---|---|
| The `engine.kv_dtype` knob is already parsed + validated | `src/serve/cluster_config.cpp:182-185`; `cluster_config.hpp:50` (`std::string kv_dtype = "bf16"`) |
| The Qwen family hardcodes bf16 and never reads the knob | `apps/dgpp_serve.cpp:472` — the only `latent_format`/`kv_format` hit inside `QwenFamily` (`:419-528`) is `return "bf16"` |
| MiMo's fp8 KV is **storage-only**, dequant in the read path | `src/kernels/mimo_attn.cu:166,180` (`store_fp8_head`), `:324,337` (`decode_fp8_piece`) |
| `latent_format.hpp` has the codecs + host oracle the tests need | `latent_fp8_row_scale` `:175`, `latent_fp8_encode` `:181`, `latent_fp8_decode_bf16` `:185`, `latent_quantize_row_host` `:262`, `latent_dequantize_row_host` `:354` |
| The "native, no-dequant" end state is hardware-feasible here | `src/kernels/fp8_gemm.cu:184` = `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`; `CMakeLists.txt:9` = `CMAKE_CUDA_ARCHITECTURES 121a` (GB10 sm_121a) |
| §3.3 / §3.5 / §3.6 / §3.8 line refs | `forward.hpp:128` ctor, `forward.cpp:52` def, `forward.cpp:182/337` shape sites, `forward.cpp:346` plan label, `layers.hpp:234` `QwenQsaCache`, `qsa.cu:770` `qsa_kv_append` — all correct |
| §3.4 members really exist to extend | `kv_pool.cpp:31` `cache_bytes`, `:62` `view`, `:85` `reset_request`, `:94` `reset_all`, `:111` `copy_block_contents` |
| The MiMo fp8-KV record the tolerance band leans on | `docs/mimo_v26_flash_plan.md:223` — "e4m3 codes with one fp32 scale (absmax / 448) per head row … 328 bytes per kv head per token against 640" |
| Keeping index/ring bf16 is justified, not lazy | index plane = 67 MiB/layer vs 1024 MiB/layer for K/V → **5.9 %** of the pool; halving it buys ~3 % of the pool for a scorer rewrite. Plan is right to defer |

### 1.1 The plan understates its own Phase A parity claim

`docs/mimo_v26_flash_plan.md:223` records that MiMo's fp8 kernel over a cache is
**bitwise** the bf16 kernel over the **dequantized rows** (`mimo_attn_test` pins
both). Phase A inherits that property: quantize-on-write + dequant-in-read means
the only divergence source is the quantization error itself, not kernel
reordering.

So plan:L71-73 / plan:L160-163 can claim **"bitwise given the same dequantized
rows; divergence only from the quantizer"** — a sharper, more falsifiable gate
than "within FP8 tolerance". Keep the tolerance band for the *transcript* gate,
but the kernel-level test can be an exact-match-against-dequantized-bf16 test.

---

## 2. Errors (fix before implementing)

### E1 — §4 gate path does not exist (plan:L157)

`gates/run-gates.sh` is **not in the dgpp repo** (`find . -name run-gates.sh` →
nothing; there is no `gates/` directory). The gate harness lives in the
deployment repo:

- runner: `~/work/dgpp-gateway/gates/run-gates.sh` (reached via `make gate`)
- api: `~/work/dgpp/scripts/serve_api_check.py`
- transcript: `~/work/dgpp/scripts/serve_greedy_transcript.py`
- prefill: `~/work/dgpp/scripts/serve_prefill_probe.py`
- eval (slow): `~/work/dgpp/scripts/serve_eval.py`

Fix: replace `gates/run-gates.sh` with "`make gate` in `~/work/dgpp-gateway`
(`gates/run-gates.sh` → `scripts/serve_*.py`)".

### E2 — §3.2 cites a GLM line as if it were Qwen (plan:L82)

`apps/dgpp_serve.cpp:540` (`kv_format_name() → "bf16"`) belongs to
**`Glm4Family`** (struct opens `:529`), not `QwenFamily`. Only `:472` is in
scope. Copying the pair literally invites touching GLM-4.7, which §0 puts out of
scope.

### E3 — §3.2 misses a required call site (would not compile)

`make_family()` already accepts the format and forwards it (`dgpp_serve.cpp:963`):

```cpp
std::unique_ptr<ServeFamily> make_family(const std::string& ckpt, int world,
                                         dgpp::LatentFormat kv_format, ...)
  if (arch == dgpp::ModelArchitecture::MimoV2) return std::make_unique<MimoFamily>(ckpt, kv_format);
  ...
  if (arch == dgpp::ModelArchitecture::Qwen4Exp)
    return std::make_unique<QwenFamily>(ckpt, rope_scaling, fp8_head_mma);   // <-- :972 needs kv_format
```

Add `apps/dgpp_serve.cpp:972` to the §3.2 file list.

### E4 — §3.1's "validated to `bf16|fp8|fp4`" is not what the code does (plan:L77)

Validation is `latent_format_from_string` (`cluster_config.cpp:184`), which also
accepts `fp8_e4m3`, `e4m3`, `nvfp4`, `e2m1`, `fp8_block`, `fp4_block`
(`latent_format.hpp:66-72`). Consequences today:

- `kv_dtype: "nvfp4"` and `kv_dtype: "fp8_block"` parse clean, then are
  **silently ignored** by the Qwen lane (no family-side check, no warning):
  `kv_format_name()` reports `bf16`, the memory plan prints "K/V bf16", and the
  pool still costs 13.81 GiB/rank. A user can believe fp8 KV is active when it
  is not.
- Only MiMo guards its accepted set (`dgpp_serve.cpp:834`).

Fix (cheap, do it first, independently of the rest): have `QwenFamily` throw on
any non-`kBf16` value, mirroring `MimoFamily`:

```cpp
if (kv_format != dgpp::LatentFormat::kBf16)   // until Phase A lands
  throw std::invalid_argument(...);
```

and/or tighten the config validator to the family-accepted set.

---

## 3. Gaps

### G1 — §3.7 names the wrong entry point and understates Phase B (plan:L126-138)

Two things are wrong with the kernel list:

1. **Naming.** The serving path calls `qsa_attn_partial`
   (`src/models/qwen/layers.cpp:807-808`, for all decode and short prefill);
   `qsa_attn_partial_gather` is the *implementation* that wrapper forwards to
   (`qsa.cu:875-881`, `async = -1`). The change lands in one body either way —
   the doc should say so, and cite both entry points (`qsa.hpp:132`, `:143`).
   The bf16-probability pin lives at `qsa.hpp:128`.

2. **Phase B is much more than a dtype swap.** Decode and short prefill have
   **no tensor cores at all** — they are warp-shuffle FMA dot products:
   `qsa.cu:274-279`, `qsa.cu:627`, `qsa_prefill.cu:109` (`__shfl_xor_sync`
   reductions). Only the long-prefill path uses MMA:
   `qsa_warp.cu:55` `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with
   `ldmatrix` (`:46-51`). So "re-express `Q·Kᵀ` and `P·V` on the FP8 MMA" means
   *building a new tensor-core lowering over `topk`-gathered rows*
   (smem staging + `ldmatrix` for non-contiguous gathered keys) in the decode
   kernel — not changing a dtype in an existing matmul.

   §3.7's Phase A wording "before the existing **bf16 matmul**" is also wrong
   for those two kernels; they have no matmul.

   Least-risk sequencing this implies: Phase B on `qsa_attn_prefill_warp`
   first (the mma scaffold already exists there), and treat the decode kernel's
   fp8 MMA as a separate, larger project.

### G2 — the numerics contracts Phase B breaks are not named

`src/kernels/qsa.hpp:126-129` pins "Probabilities round to bf16 for the V
accumulation, `l` stays unrounded (the DSA pin)". `src/kernels/qsa_prefill.cu:3`
pins the rounding order against decode. `qsa.hpp:164-165` already documents
warp-prefill as *tolerance-equal* to `partial + dsa_attn_combine` (a precedent
the plan can cite for its band).

Phase B moves `P` to E4M3 and breaks those pins. The plan should name the tests
that hold them — `tests/cuda/qsa_test.cu`, `tests/cuda/qwen_full_attn_test.cpp` —
so they are updated deliberately rather than discovered as red CI. Same for the
"every gather form is bitwise every other" comment (`qsa.hpp:141-143`).

### G3 — the memory math is 1.86×, not 2×, and the doc should carry the real numbers (plan:L10, plan:L104-106)

Measured from the live plan line (13 layers × 1 kv-head/rank × dim 256,
`kpool=4`, `idx_dim=128`, `kv_capacity=1048576`, `max_concurrency=4`):

| | bf16 (today) | fp8 |
|---|---|---|
| K/V per layer | 1.000 GiB | 0.500 GiB |
| scales per layer (2 × `slots·kv_heads·4 B`) | — | 0.008 GiB |
| index per layer (unchanged) | 0.0625 GiB | 0.0625 GiB |
| **per layer** | **1.0625 GiB** | **0.5703 GiB** |
| **13 layers, per rank** | **13.81 GiB** | **7.41 GiB** |

Source: `log/deployments/c1c3cdef3b168073/serve_r1.log` — "kv cache pool (K/V
bf16, compressed index keys, rings) **13.81 GiB**" (matches
`QwenKvPool::cache_bytes` exactly).

- K/V planes alone: **1.97×**
- whole pool: **1.86×**
- freed: **6.40 GiB per rank / 12.8 GiB across both Sparks** → headroom
  8.4 → **~14.8 GiB free**

Two smaller text fixes in the same section:

- plan:L105-106 reads "+ `·4` for the two scale planes" — should be
  **2 planes × `token_slots · kv_heads · 4 B`** (8 B per slot-head).
- plan:L112 `validate()` is a **file-local free function** in an anonymous
  namespace (`kv_pool.cpp:11`), not a member method.

**Crisp framing for the doc:** *fp8 at 2 000 000 pool tokens costs what bf16 at
1 048 576 costs today* → with 4 requests capped at the native 256K context,
that is **8 slots at 256K instead of 4** (no YaRN needed), or push
`prefix_cache_gib` 42 → 48. Note the pool is per-rank, so the win is counted
twice on a TP=2 lane.

---

## 4. Not a gap (but worth one sentence in §0)

`QwenFullAttnLayer` (`full_attn_decode` / `full_attn_prefill`,
`src/kernels/full_attn.cu:482,513`, called from `layers.cpp:955,959`) reads a
same-shaped `k_cache`/`v_cache` — but it is **Qwen3.5-only**
(`src/models/qwen/model35.cpp:665`) and rides a *different* pool struct
(`layers.hpp:325`). It is correctly out of scope; say so explicitly so nobody
"fixes" it, and so §0's "scope: `src/models/qwen/`" isn't read as covering
`model35.cpp`.

---

## 5. Suggested edit list for the plan

1. §0: add the `QwenFullAttnLayer`/Qwen3.5 exclusion sentence; restate the win
   as 1.86× pool / −6.40 GiB per rank (not a clean 2×).
2. §0.2 + §4.2: Phase A parity = bitwise-vs-dequantized-bf16 at kernel level
   (MiMo precedent), tolerance band only for the transcript gate.
3. §3.1: drop "no change" → add the Qwen-family `kBf16`-only throw as
   pre-work; note the alias-permissive validator.
4. §3.2: cite `dgpp_serve.cpp:472` only; add the `make_family` site (`:972`).
5. §3.4: fix the scale-plane formula; `validate()` is file-local.
6. §3.7: cite `qsa_attn_partial` (`qsa.hpp:132`) + `_gather` (`:143`) as one
   body; state that decode/short-prefill are shuffle-FMA (no MMA today) so
   Phase B needs a new tensor-core lowering there; propose doing Phase B on
   `qsa_attn_prefill_warp` first; drop "existing bf16 matmul".
7. §4.1: name `tests/cuda/qsa_test.cu` + `qwen_full_attn_test.cpp` as the
   rounding-order pins to update.
8. §4.2: replace `gates/run-gates.sh` with `make gate` in `~/work/dgpp-gateway`.
9. §6 (DoD): add the README row (`README.md:451` says "Qwen, GLM-4.7 and
   DeepSeek K/V caches remain BF16") + `docs/operations.md`, and a boot-plan
   assertion that the pool line actually halved.
