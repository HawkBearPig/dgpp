# Issue #4: preserve trained EOS for PLE

An independent audit of the exact 261,290-token forced prefix found serving
replaced Qwen's trained EOS 248044 with the generation stop list's first ID,
248046. PLE used the replacement for n-gram padding and segmentation; this
changed 40 lookup IDs in the exact diagnostic prefix, including the boundary
just before the answer. The patch separates generation stop policy from model
configuration and makes the serving family EOS accessor const. Both generation
terminators remain active for decoding and grammar handling.

The correction does **not** fix issue #4. The unchanged original 261,120-token
request still returns 5/6, byte-identical to the baseline wrong answer
`val_5dac9ed720abddaf`. Native prefill completed with 184 output tokens, zero
cached tokens and normal stop. The planned bounded run was skipped after that
negative causal result. Production was restored and independently verified.

Release build and the new EOS regression passed. After building all selected
host targets, the complete `ctest -L host -LE checkpoint` run passed 22/22
with no failures or unbuilt targets. This is a separate
confirmed serving bug, not a claimed resolution of the retrieval issue.

The local raw evidence retains the tested binary and full host logs. Additional
spread/tail retrieval controls were prepared but not run, because the original
regression still fails.

The defect occurs before family construction of sessions/layers: the serving
entry point assigns generation stop IDs into the family's model configuration.
For this checkpoint `config.json` has `[248044]`, whereas
`generation_config.json` has `[248046,248044]`. Qwen's PLE hash/session history
use the first model EOS. This substitutes the chat terminator for the trained
padding/reset token. Decoding still needs both generation stop IDs.

The serving-family getter is now const, preventing this configuration mutation.
The independent stop list feeds scheduler, grammar vocabulary and HTTP serving.
The regression checks the real differing ID order, missing-generation fallback,
and an explicitly empty generation stop list. No kernel arithmetic changes are
part of this fix.

Validation commands (release configuration):

```sh
cmake --build build-release -j 4 --target dgpp_serve_app unit_tests http_server_test serve_test fabric_serve_test scheduler_test roster_check qwen_yarn_fixture_test
ctest --test-dir build-release -L host -LE checkpoint --output-on-failure
```

The TP2 run used checkpoint `fc694b54fb0174e0913e6adf86691ef85a4ead47`,
four slots, native prefill, BF16 KV, FP8 dense, MTP depth one, and the original
261120-token request with 1024 output reservation. The tested binary SHA256,
source base and request SHA256 are in `tp2-build.json`; `native-verdict.json`
retains the negative causal result. Production's original four-node binary and
configuration were restored and an inference smoke check returned `OK`.
