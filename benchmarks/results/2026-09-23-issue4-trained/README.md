# Issue #4: combined trained-settings control

The unchanged original request was run on fresh TP2 with trained EOS semantics,
checkpoint BF16 dense weights and the independently checked NVFP4 activation
round-trip. It still returned 5/6 and the original wrong value
`val_5dac9ed720abddaf`, with 261,120 prompt tokens, 184 output tokens, no cached
prompt tokens and a normal stop. Bounded prefill was not run after this failure.

This rules out the tested combination as a sufficient correction. It does not
establish equivalence to vLLM's expert tensor-core arithmetic or recurrent
implementation. The source diff/header, binary identity, exact request, live
metrics, response and logs are retained under raw/campaign. Production was
restored with exact original binary/config identity and an independent smoke.

The isolated EOS bug is preserved in the adjacent issue4-eos record. The W4A4
round-trip remains a diagnostic, not a proposed production fix.
