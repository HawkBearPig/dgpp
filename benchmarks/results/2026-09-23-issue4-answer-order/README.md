# Issue #4: answer ordering diagnostic

The clean baseline binary was used, SHA256
`b6afe5546afb687e4f9adea276625b2cbff2ac38ba6dc60eab55879a50ddba47`.
Both tests ran in fresh TP2 worlds, as their first generation, with BF16 KV
and FP8 dense settings. These are altered-input diagnostic controls, not a
proposed fix or an original-request success.

1. Keeping all original 261120 prompt tokens, force the first answer key
   through `{"key_0769e0226c63":"val_` (261138 total tokens). Completion
   returns `val_905_4_wrong`, the immediately adjacent decoy's value. There
   are no earlier generated associations, so their presence is not necessary
   for an incorrect association. MTP/prefix caching are disabled for this
   completion control.
2. Reorder only the requested-key list to put that key first, retaining all
   ledger text and instructions. Native rendering still gives 261120 tokens.
   Original MTP/prefix settings are retained. The model chooses all five
   decoy values and returns null for the absent key: 1/6, 148 completion tokens,
   normal stop, zero cached tokens. This demonstrates answer/query-order
   sensitivity on this fixture. It does not establish an implementation defect.

`input-parity.json` binds token counts and the forced prefix. The two verdicts
retain complete scored outputs. Production's binary and configuration were
restored and verified with successful inference. The corrected standalone GR
comparison was also run serially during the idle hardware window; those results
are copied into the adjacent gr-precision record and do not affect this server.

The campaign recorded the current worktree diff for provenance, but the server
was the retained clean baseline executable, not a build of that unrelated diff.
