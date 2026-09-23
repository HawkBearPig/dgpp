# Issue #4: reference expert-finalization control

Executed with the original request twice in one TP2 world. Both answers remain
5/6, selecting `val_5b9d2c3d7a95b986` then `val_5dac9ed720abddaf` for the failed
key (190 and 184 output tokens). All 261120 native prompt IDs match; prefix
caching is disabled. Production was restored and independently verified.

The one-line reference overlay disables fused MoE finalization, which otherwise
permits BF16 atomic accumulation. It retains the previous trace instrumentation.
All **340 full first-chunk fields are now bitwise identical** across the two
requests, where the traced baseline first differed at layer-0 MoE output.
This supports fused expert finalization as the source of that initial variation.
It does not remove all reference variation or fix DGPP issue #4.

Both ranks first show a later sampled difference at chunk start 2048,
`layer3_attn_out`, sampled token 4095: 556/2560 different values, relative L2
0.0020684484, maximum absolute difference 0.0009765625. Later chunks retain only
the last row, so this is the first sampled difference, not proof that all earlier
rows in that chunk match. Full results: `rank-comparison.json`.

A separate FlashInfer cache root retained only the original 21 GEMM1 entries;
all remain unchanged. GEMM2 retuned because disabling finalization changes its
eligible tactic list. New cache SHA256:
`4b34b1169b71e0fb3f8458c3636cf59bc9b7c569999e6a3606407fb961b1d3f8`.
The original reference cache was not changed.

Instrumentation changes scheduling and the traced responses differ from the
uninstrumented failure/success pair. The findings localize sources of reference
variation; they do not establish the cause of DGPP's consistent wrong answer.

After verified production restoration, slow SFTP capture collection was replaced
by a successful tar stream. The campaign's exit 1 is confined to the intentionally
interrupted post-restoration SFTP step; `collection.json` records recovery and
completed both-rank comparisons. Raw failure/command receipts are retained.
