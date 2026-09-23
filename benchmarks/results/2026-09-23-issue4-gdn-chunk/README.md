# Native chunked GDN prototype — original request still fails

The 64-token WY block prototype passed all 24 frozen-input comparisons with the
pinned reference and all five executed GPU tests. Maximum relative L2 errors
were 0.001332 for output and 0.000551 for final state, within the predefined
0.002 / 0.001 budgets. The standalone deterministic regression output agrees
bitwise with the pinned reference; the original recurrent kernel fails this
operator regression. These results validate the tested operator policy, not
retrieval accuracy.

The unchanged original TP2 request still scores **5/6**, returning
`val_7265273163533973` for `key_0769e0226c63`, with 261120 prompt and 190
completion tokens in 299.078 seconds. That wrong value is present in another
record, `key_07600e2d7f8f`. The bounded follow-up was skipped. The exact original
four-node production binaries and configuration were restored and independently
verified with an inference smoke test.

The worktree starts from the separate EOS fix. Decode and speculative snapshots
retain the original recurrent path. Dense-weight policy, request settings and
all input token IDs remain unchanged. This is **not a retrieval fix** and is
not proposed for production.

The frozen-input configuration probe holds all inputs fixed. Six of seven FLA
kernel configuration changes leave output and state bitwise unchanged; changing
the 64x64 triangular-inverse merge introduces output relative L2 9.86e-6 and
state relative L2 8.98e-6. This does not identify configurations selected in the
earlier successful and failing full-model worlds. Unsupported launch shapes
were recorded and excluded. A KKT precision probe supports TF32 truncation of
the beta*K operand; adopting that contract repairs four prototype operator
mismatches, but does not repair the original request.

Two earlier operator attempts are retained locally under `raw/attempt1-fp32-kkt`
and `raw/attempt2-fixed-reference`. Both failed operator checks and restored
production before any full-model candidate request. `build-provenance.json`
and `diagnostic.patch` describe the five-test binary actually used in the
completed third campaign. A subsequently built sixth fixture test has not yet
run on the GPU and is not included in that validation claim.
