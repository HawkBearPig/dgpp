# Real-fixture MoE audit — prepared, not run

Read-only capture on master c6ca191: three rows in the first and final native
chunks of the retained 261290-token forced prefix, all 48 layers, both ranks.
Capture includes actual routed input, logits, selected IDs/weights, segmentation
mapping, gate/up/activation/down intermediates, the routed sum, shared-expert
intermediates and local/folded MoE outputs. No kernel or loader arithmetic changes.

The original diagnostic response, logprobs and usage must exactly match the
retained clean executable before interpreting the captures. Each model input
is independently replayed from checkpoint weights using FP64 matrix products,
explicit BF16 staging and an independently decoded NVFP4 format. Shared dense
FP8 conversion is reconstructed from the sliced BF16 checkpoint in NumPy.

Local stage checks use captured preceding inputs; the complete expert-chain
check carries its own intermediates using captured routing decisions. Routing
logits and selection are separately audited. This cannot validate earlier
layers that produced the captured MoE input, nor unsampled rows.

The existing qwen_moe_test expert budget is recorded, not chosen to fit this
fixture: zero hard violations over 12 BF16 ulps, fewer than 2% over four ulps,
a 2%-RMS absolute floor, and relative L2 below 0.004. Exact routing/segmentation
and rank-fold invariants are checked independently.

Host checks compare all 256 E4M3 encodings, BF16 rounding and partial 128x128
block FP8 quantization against Torch. Production restoration and identity checks
execute before collecting the peer captures.
