# Marlin first-chunk expert probe — grouping variation localized

Two fresh full-model traces first differ at layer-0 MoE output with identical
preceding captured inputs and identical GDN launch selections. This bounded
probe sends the exact first 2048 original token IDs directly, twice, and checks
that the layer-0 MoE input matches the full original request. It adds read-only
router, shared-expert and Marlin intermediate captures, and exports the actual
prepared Marlin operands for isolated replay. The short request is diagnostic;
it cannot provide a retrieval accuracy verdict. GDN launch configurations are
pinned to those measured in the first full trace, and state remains BF16 auto.

Both rank inputs match the full original request exactly. Across the two short
requests, hidden states, router logits, expert IDs/weights, shared-expert output
and padded counts repeat exactly. The first numerical difference is the routed
gate/up projection. Token grouping changes in 19340 valid slots on rank 0 and
19297 on rank 1; this is not a comparison of uninitialized padding.

Isolated GPU replays use the captured prepared weights and inputs. Each of four
fresh processes (two per rank) runs ten automatic-grouping and ten frozen-grouping
replays. Automatic grouping produces ten distinct outputs in every process.
Holding the captured grouping fixed makes every gate/up, activation, down and
routed-sum output bitwise identical to the original capture, including across
process restarts: all 40 frozen replays pass. The kernels already request FP32
reduction and disable atomic accumulation.

This identifies grouping order as a causal source of reference expert variation.
It does not establish the cause of the long retrieval failure. DGPP already uses
stable flattened token/slot ordering within each expert. A canonical reference
grouping control is validated separately and is being tested on the full request.

The capture and replay campaigns each restored the exact original four-node
production binaries/configuration and passed an inference smoke test. Executed
results are in `same-world-comparison.json`, `replay-summary.json`,
`replay-receipt.json`, `restoration.json` and `replay-restoration.json`.
The large operand bundles and raw captures remain local under ignored `raw/`.
