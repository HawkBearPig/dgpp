# FP32 reference state — repeatable original failure

Both unchanged original requests score **5/6**, returning the original wrong
`val_5dac9ed720abddaf` with 261120 prompt / 184 completion tokens. Full text and
usage match each other and the preceding canonical BF16-state reference exactly.
Wall times are 340.328 and 335.063 seconds. The exact original four-node production
binaries/configuration and inference smoke test passed after restoration.

This control adds `--mamba-ssm-cache-dtype float32` to the canonical-grouping
reference. It retains deterministic QSA, canonical Marlin grouping and the exact
pinned GDN launch configurations from the BF16-state world. Grouping operator
validation is reused from the canonical control, not rerun.

The first chunk's captured values match the BF16-state control exactly on both
ranks, including all non-state tensor bytes. The three initial zero-state tensors
per rank differ only in storage dtype. The first numerical difference on both
ranks is layer-0 initial state at position 2048, exactly the intended cache boundary.
The next state equals the preceding FP32 result bit for bit in all 12 checked
rank/request/layer handoffs. This separates the storage effect from earlier
projection, expert-grouping or launch-selection changes.

Across repetitions, all **267480 captured fields** across both ranks repeat
exactly, with no token drift or missing final fields. All **534960 saved payloads**
pass shape, length and SHA256 checks (6701434528 bytes). Against BF16 state, each
rank has 4343 byte-identical fields, three equal-valued initial states with different
dtypes, and 129394 differing fields. The answer nevertheless stays unchanged.

These results establish that correcting reference state storage alone does not
repair this fixture. DGPP already uses the checkpoint-declared FP32 state; this
control provides no reason to lower its precision or promote the failed GDN prototype.
See `comparison.json`, `versus-bf16-state.json`, `state-storage.json`,
`capture-integrity.json`, `response-parity.json` and `restoration.json`.
