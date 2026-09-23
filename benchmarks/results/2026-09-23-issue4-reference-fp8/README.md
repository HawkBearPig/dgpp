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
stopped production at 15:12 UTC; the reference is loading. Accuracy results are
pending. The campaign restores production in its cleanup path.

A CPU-only check instantiated the pinned reference configuration class for both
checkpoints. Effective routing normalization, PLE seed and convolution width
agree (`true`, `1234`, `4`), despite omitted optional metadata in the FP8 config.
`effective-config-parity.json` records the values. The PLE consumer supplies seed
1234 and GDN uses `linear_conv_kernel_dim`; neither omission changes execution.
