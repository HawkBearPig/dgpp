# Canonical Marlin grouping — stable operators; original request still fails

Frozen real-operand replays identify nondeterministic token grouping as a source
of Marlin expert output variation. This diagnostic control retains the original
expert counts, block sizes, padding, routing and arithmetic kernels, but places
flattened token/slot IDs in ascending order within each expert. It supports the
observed TP-only case; expert-parallel mappings are explicitly rejected.

All 60 GPU grouping cases pass an independent CPU enumeration, including small
batches, padding choices and the actual 2048-token/512-expert shape. Canonical
grouping gives bitwise-identical expert outputs in 40 replays across four fresh
processes and both captured rank operands. The frozen-capture positive control
also passes all 40 replays. Canonical versus original grouping changes 32 and 49
of 5242880 routed-sum elements on ranks 0 and 1 respectively (relative L2 errors
5.05e-6 and 6.65e-6). See `alignment-tests.json` and `unit-summary.json`.

The unit campaign restored production and passed its smoke test. The full TP2
campaign completed two unchanged original requests. It retains BF16 auto state,
pins the previously measured GDN launch choices per rank, and applies deterministic
QSA plus canonical expert grouping. Both score 5/6, returning the original wrong value `val_5dac9ed720abddaf`
with 261120 prompt and 184 completion tokens (316.779 and 324.704 seconds).
Full text and usage repeat exactly and match the preceding uncontrolled-grouping
trace. Production restoration passed. Both ranks repeat all 133740 captured fields
exactly, with no token drift or incomplete tail. All 534960 saved payloads pass length, shape and SHA256 validation
(6682560160 bytes total; no orphan payloads). Single-configuration GDN autotuners
bypass cache selection, so their cache receipts are empty; the allowed launch
choices are fixed by the verified helper and `pinned-gdn-configs.json`.
There is no retrieval fix.
