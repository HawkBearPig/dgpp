# Explicit FP32 reference state — running

The observed reference recipe stores GDN state in BF16 between calls despite
FP32 within-call accumulation. DGPP uses the FP32 state declared by the checkpoint.
This control adds `--mamba-ssm-cache-dtype float32` to the canonical-grouping
reference campaign. It retains deterministic QSA, canonical Marlin grouping and
the exact GDN launch configurations measured in the earlier BF16-state world.

Grouping implementation and unit-summary hashes match the validated
[canonical control](../2026-09-23-issue4-reference-canonical-moe/README.md).
Its operator validation is reused, not rerun. Two unchanged original requests
and state/activation traces test the cache-storage difference; there is no
accuracy verdict yet. This is not a proposed change to DGPP state precision.
