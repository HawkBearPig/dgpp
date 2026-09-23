# Issue #4: reference tactic ablation

Prepared, not yet executed. This uses the exact pinned reference image, source,
checkpoint, input and engine settings, changing only kernel configuration to
`enable_flashinfer_autotune: false`. The pinned source skips FlashInfer cache
loading/tuning in that configuration and relies on backend heuristics instead.
This tests whether tactic selection affects the original request's correctness.
It is not a DGPP implementation change or a proposed production workaround.

The campaign verifies all native input IDs and restores production in finally.
