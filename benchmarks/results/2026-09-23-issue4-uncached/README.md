# Checkpoint rebuild control — executed, 5/6

With all weights built directly from the verified original checkpoint, the
unchanged original request returns the same wrong `val_5dac9ed720abddaf`.
Assistant text and usage are identical to the cached EOS baseline: 261120
prompt tokens, 184 output tokens, zero cached tokens, normal stop.

Both ranks' retained logs have no resident image open/load summary. The sole
diagnostic patch forces the loader's image directory empty, bypassing both
reads and writes. Existing cache files are unchanged. This intervention did
not change the original failure; it does not invalidate the baseline result.

`build-provenance.json` identifies the retained binary, EOS base commit and
sole diagnostic patch. Production restoration and identity verification
execute in `finally`.

Production restoration completed with original binaries/configuration on all
four nodes and an `OK` inference smoke check.
