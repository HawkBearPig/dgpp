# Frozen GDN algorithm comparison — completed

Sixteen one-head cases from ranks 0/1, layers 0/16/32/46, first 2048 and final
1194 prefill rows. Each retains actual convolved Q/K/V, raw gates, parameters,
starting/ending state and output. These are selected from the previously
validated full-history capture, with input hashes recorded.

The pinned reference image runs its recurrent operator with in-kernel FP32
Q/K normalization, its recurrent operator with normalized BF16 Q/K, and its
active chunked prefill operator with those same normalized BF16 inputs.
All three use identical raw inputs and starting states. This separates the
normalization storage boundary from the chunked algorithm's arithmetic.

The recurrent FP32 path is compared with DGPP's captured output/state as a
control. The chunk/recurrent comparison does not assert that either policy
is the cause of the full retrieval error. Compilation-inclusive timings are
not production performance measurements.

The isolated GPU container runs on idle node 13 after production is stopped,
before the reference TP2 world starts. Cleanup verifies that the container
is removed even on timeout; the enclosing campaign restores production.

The first harness attempt supplied no state-index tensor to the serving
recurrent kernel's in-place mode and failed during Triton compilation before
producing data. Its source/log are retained under `raw/failed-no-indices*`.
The corrected harness supplies a padded state pool and explicit valid slot
indices for every token. No numerical conclusion comes from the failed attempt.

All sixteen cases completed with finite outputs/states. See `summary.json`
for maxima and `comparison.json` for every case. The independent reference
recurrent control closely matches the original DGPP capture. The chunked
algorithm differs after holding normalized inputs and initial state fixed.
The isolated container was removed before the TP2 reference world started.
