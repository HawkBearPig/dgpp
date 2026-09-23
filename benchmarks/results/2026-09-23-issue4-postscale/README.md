# Issue #4: closer reference arithmetic control

Executed but **invalid as an accuracy control**. The server loaded all 49 layers
from a pre-existing resident weight image, bypassing the new loader conversion.
The activation quantizer passed 96 cases / 31457280 scalars (zero numerical
mismatches; 1473 signed-zero differences), but that does not validate integration.
The response was malformed text, hit the 1024-token limit, and the strict scorer
raised a JSON error. It is not evidence against correctly integrated W4A4
arithmetic. Production was restored and independently verified.

A fresh-checkpoint baseline control is being prepared before rerunning this
change. The existing image cache is preserved. `invalid-control.json` records
the invalidating condition.

The Diagnostic branch `investigate/issue4-postscale` starts
from the independently tested EOS correction. It combines checkpoint BF16 dense
weights, FP32 gated-residual intermediates, and NVFP4 activation rounding.

Unlike the earlier activation control, activation block values retain their
exact E2M1 × E4M3 product in BF16; the input global scale joins the weight global
scale in the dot-product epilogue. SwiGLU uses one final BF16 rounding before
activation quantization, matching the inspected pinned reference source.

This still uses DGPP's BF16 tensor-core dot products and expert reduction order,
not native NVFP4 tensor cores or an exact vLLM clone. It is a diagnostic
combination, not an established fix. If it changes correctness, further
ablation and boundary comparisons are needed before proposing production code.

The first build invocation mistakenly named the output file instead of the
CMake target and did no work. It was corrected to `dgpp_serve_app` before any
GPU test; the no-op executable is not a tested candidate.

The cache key includes checkpoint/configuration/loader-format identity, not every
local source edit. This diagnostic changed cached tensor contents without changing
the loader-format identity. `paths.resident_cache: ""` means the default cache,
not disabled caching. Earlier quantizer and GR-only controls do not change loader
payloads, so this particular invalidation does not apply to them.
