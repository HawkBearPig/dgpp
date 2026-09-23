# Independent FP8 checkpoint reference — running with the authorized cluster window

DGPP's earlier FP8-checkpoint control also failed the original association. This
comparison uses that same cached `Qwen/Qwen3.8-Flash-Next-FP8` revision
`236dfdf285828023ca3bcd3f37366c58a3469b13` in the pinned independent reference.
Only the request's model name changes; every input token must still match.
The earlier record verifies all 144 files on both nodes and exact equality of
common non-expert language tensors and PLE payloads with the NVFP4 checkpoint.
No checkpoint download or conversion is involved.

The reference selects its existing Triton W8A8 expert path. Its stock loader
losslessly refines the 128-by-128 scale grid to 64-by-64 to accommodate TP2's
320-wide intermediate partition. A read-only first-layer check compares four
experts' actual loaded gate/up/down bytes and refined scales against independent
checkpoint slices on each rank before accepting an accuracy verdict (48 checks).
This is sampled loader validation, not a complete independent model execution.

The reference retains explicit FP32 GDN state, pinned GDN launch choices and
deterministic QSA. The previously validated canonical grouping helper is applied
to Triton's common grouping path; small decode batches use its stock deterministic
naive assignment. Grouping validation is reused; the earlier Marlin operator
replay is not a validation of FP8 expert arithmetic.

Two full requests will test accuracy and captured repeatability. This changes
expert weight precision and activation/backend arithmetic together; it cannot
identify a particular NVFP4 operator defect or constitute a fix for the original
NVFP4 request. Exact production restoration is required after the campaign.

The first launch aborted at its idle-service guard before stopping production or
starting a reference world. It sent no test inference requests. `attempt1.json`
records the exclusion; production remained live. The user subsequently authorized
stopping the cluster for testing. The retry passed both recorded idle checks and
stopped production at 15:12 UTC. This attempt was excluded because the diagnostic
checker incorrectly asserted FP32 checkpoint scales before the first expert
computation. Production restoration passed. The corrected retry is starting;
accuracy results remain pending.

A CPU-only check instantiated the pinned reference configuration class for both
checkpoints. Effective routing normalization, PLE seed and convolution width
agree (`true`, `1234`, `4`), despite omitted optional metadata in the FP8 config.
`effective-config-parity.json` records the values. The PLE consumer supplies seed
1234 and GDN uses `linear_conv_kernel_dim`; neither omission changes execution.
The small CPU source/output receipts are also published in `config-receipts/`;
`config-parity-provenance.json` hashes their original local paths.

## Excluded checker failure and correction

`attempt2.json` records the diagnostic-only abort. All 12 sampled checkpoint scale
tensors are BF16; the stock reference allocates FP32 runtime scales. The corrected
checker requires those actual types and compares exact FP32 promotions of the
BF16 checkpoint values. It still checks raw FP8 weight bytes and refined scale
coordinates exactly. The CPU preflight independently decodes BF16 bits, expands
full 128-by-128 scale blocks, shards them and regroups into 64-by-64 blocks. All
48 checks pass, and deliberately corrupted scales are rejected on both ranks.
The campaign requires this preflight and its checker hash. The aborted attempt
also confirms all 17 captured fields before the first expert computation match
the NVFP4 FP32-state control on both ranks. No retrieval verdict is claimed.
