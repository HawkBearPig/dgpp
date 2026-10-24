# PR #101 follow-up review

The implementation and current measurements are documented in
[qwen_fp8_kv_plan.md](qwen_fp8_kv_plan.md). This replaces the pre-implementation
review, whose line references and planned phases no longer describe the code.

The follow-up resolves these issues:

- The FP8 append kernel originally assigned one head to each of eight warps,
  leaving scales uninitialized above eight KV heads. Warps now iterate over
  heads; regressions cover nine and sixteen heads.
- The attention test used a query-only row stride for the interleaved query/gate
  buffer. It now uses the full row stride.
- Storage-only attention widened cached K/V to BF16 for both products. Native
  E4M3 QK/PV is now independently selectable for prefill and decode, with
  per-token V scales applied inside the reduction. The accurate storage-only
  path remains available and uses hardware conversion.
- A 6.25% FP8 element-rounding figure was presented as an attention-output error
  guarantee. It is not such a guarantee. Regression thresholds and measured
  model-logit comparisons are now identified as empirical evidence.
- The original validation did not pin exact dequantization, scale-plane block
  copies, independent native settings, or native graph replay. Those now have
  regression coverage, alongside the recorded performance/accuracy limits.

Native FP8 remains opt-in. Broader task-quality, long-context accuracy, and
matched end-to-end performance measurements are still needed before promoting it to a
model-family default.
