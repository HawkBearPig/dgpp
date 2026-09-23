# FP8 checkpoint control — executed, 5/6

The unchanged original text/token sequence also fails with the FP8 expert
checkpoint: `key_0769e0226c63` returns `val_51bf3e137245e273` instead of
`val_aed39758a1abf065`. The other five checks pass. There are 261120 prompt
tokens, 174 output tokens, zero cached tokens, and a normal stop.

Higher expert weight precision alone does not repair the failure. This is not
a proof that either implementation is correct or that quantization is irrelevant.

During rank-0 weight loading, writing the new optional resident image exhausted
disk at layer 42. The loader catches this cache-only error, destroys the image
handle, and continues loading weights from the checkpoint; device weights are
unmodified. The partial file created by this campaign was deleted, recovering
51 GiB. See `cache-cleanup.json` and the retained server log. Rank 0 loaded
from the checkpoint (zero prior cached layers).

Wall time: 231.3041544709704 seconds.

## Setup

Executed after the reference and invalidated arithmetic control. Used the retained EOS-corrected executable, without
arithmetic diagnostics, on the cached `Qwen/Qwen3.8-Flash-Next-FP8` snapshot
`236dfdf285828023ca3bcd3f37366c58a3469b13`. Both nodes have all 131 shards.
The original request changes only its served model name. Native template/token
identity and full checkpoint content hashes are checked before inference.

The engine settings otherwise match the EOS NVFP4 run: TP2, native prefill,
BF16 KV, converted FP8 dense projections, checkpoint BF16 auxiliaries, mmap
n-gram table, four slots, 262144 capacity, MTP depth one and full admission.

This changes the checkpoint and is not a fix of the original NVFP4 request.
It tests whether higher precision weights improve this retained retrieval case;
a positive result alone cannot identify an individual kernel defect.

Preflight evidence is complete: all 144 snapshot files (including 131 weight
shards) content-hash verified on both nodes, with identical results. All 1096
common non-expert language-model/MTP BF16/F32 tensors match byte-for-byte
(8805135618 bytes per checkpoint), as do all 128 FP8 PLE table payloads
(51200245760 bytes per checkpoint). Vision tensors were not included in the
cross-checkpoint comparison. Native text and all 261120 token IDs match.

Production was restored with all four original binary identities, exact resolved
configuration and a successful inference smoke check.
