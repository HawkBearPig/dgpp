# Deterministic W4A4 reference without tracing — running

This repeats the deterministic W4A4 reference with boundary captures and read-only
top-k probes removed. The QSA helper and QSA overlay are byte-identical to the
successful W4A16 reference. The CUTLASS expert overlay still disables fused
finalization and uses the same retained post-finalization-change tactic cache.
Both ranks must pass the cache hash check before stopping production.

Two unchanged original requests run in one fresh TP2 world. Compared with the
prior deterministic W4A4 campaign, only tracing/probes are removed. Compared
with the successful W4A16 world, the expert backend and its associated staging
and activation quantization differ. This does not isolate activation quantization
from every other backend arithmetic difference.

Production restoration and independent binary/config/inference checks run in finally.
