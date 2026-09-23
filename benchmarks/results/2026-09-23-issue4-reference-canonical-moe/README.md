# Canonical Marlin grouping — operator validation passed; original request starting

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
campaign is running two unchanged original requests. It retains BF16 auto state,
pins the previously measured GDN launch choices per rank, and applies deterministic
QSA plus canonical expert grouping. Stable operator outputs alone do not establish
a retrieval fix; the full-request verdicts are pending.
