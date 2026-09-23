# Fresh Marlin world trace — original failure reproduced

The unchanged original request scores **5/6**, returning the original DGPP wrong
value `val_5dac9ed720abddaf`, with 261120 prompt / 184 completion tokens in
339.832 seconds. Full assistant text and usage match the preceding untraced
Marlin failure exactly. All input token IDs match. The exact four-node production
binaries/configuration and inference smoke test passed after restoration.

Read-only captures retain full tensors for layers 0–3 in the first two chunks,
last rows otherwise, and prepared GDN inputs/core outputs throughout prefill.
Each rank has 312 forward calls and 133740 captured fields. Every binary payload
matches its recorded length and SHA256 (3341280080 bytes total). The second fresh
trace is running to compare numerical boundaries and actual launch selections.
Instrumentation can affect scheduling; output parity above limits that caveat.

**New state-storage observation:** the first three GDN layers on both ranks
return FP32 state but supply BF16 state to the next call. Every next-state value
matches BF16 rounding of the preceding result exactly. Relative L2 differences
from this store range from 0.001566 to 0.001788 across these six states. The
pinned reference's `MambaStateDtypeCalculator` resolves `auto` to the model dtype,
BF16. The supplied recipe also uses `auto`. The checkpoint declares
`mamba_ssm_dtype=float32`; DGPP's config parser explicitly requires FP32 state.
This establishes an additional policy difference, not its retrieval effect.
The frozen GDN operator comparisons used supplied FP32 DGPP state and do not
cover this whole-model cache conversion.

Runtime GDN configurations are retained for both ranks. They differ between
ranks, including KKT block shape and triangular-inverse warp count. This does
not establish which configurations ran in the earlier successful Marlin world.

Six independent/focused native GDN GPU tests passed before this reference world
started, including the new pinned-reference fixture regression. This does not
change the native prototype's failed long-request result.

After successful inference and verified restoration, slow recursive SCP was
replaced with a streamed tar transfer. The campaign's only nonzero exit is that
explicitly interrupted post-restoration copy; `peer-copy-stream.json` records
successful replacement, and `capture-integrity.json` verifies every payload.
