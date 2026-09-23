# Issue #4: routed-expert activation precision diagnostic

This experiment isolates activation quantization on the pinned NVFP4 checkpoint.
The baseline uses W4A16 experts. The checkpoint was calibrated for W4A4, and the
reporter's successful vLLM control uses FLASHINFER_CUTLASS NVFP4 experts. The
[separate EOS correction](../2026-09-23-issue4-eos/README.md) is **not** included
in this diagnostic, so the only forward arithmetic change is the routed experts'
activation round-trip.

The diagnostic rounds gate/up inputs and post-SwiGLU down inputs through groups
of 16 E2M1 values with E4M3 block scales and the checkpoint's calibrated global
input scale. It keeps the router and shared expert inputs unchanged. All 512
experts' input scales were checked equal within each layer/projection, and gate
and up scales were checked equal. `input-scales.json` and the diagnostic header
retain those values. The header is deliberately specific to this checkpoint;
this is not a production loader implementation.

The GPU quantizer matched an independent CPU Torch oracle on 31,457,280 values
across 96 calibrated scales, including frozen real activations, broad amplitude
ranges and zero groups. In-place width-320 and out-of-place width-2560 execution
matched bit for bit. 1,473 differences from the oracle were solely signed-zero
encodings; no nonzero value differed. The arithmetic follows the pinned upstream
NVFP4 emulation source, whose URLs/hashes are retained under raw/.

This emulates activation quantization before DGPP's existing expert dots; it is
not a claim of bitwise equivalence to FlashInfer NVFP4 tensor-core arithmetic.
The original-request test still fails 5/6. Activation quantization changes the
wrong association to `val_51bf3e137245e273`, with 189 output tokens, zero cached
prompt tokens and a normal stop. The bounded run was skipped. This intervention
does not by itself explain or resolve the failure. `campaign.py` restores
production on exit with independent binary/config identity and an HTTP smoke.

Precision limitation: this diagnostic rounds activations to NVFP4, then restores
them to BF16 before the W4A16 dot product. Native W4A4 hardware applies the
combined global input/weight scale after the dot product. The two paths are not
arithmetically identical, so this negative result does not exclude an exact
native W4A4 implementation.
