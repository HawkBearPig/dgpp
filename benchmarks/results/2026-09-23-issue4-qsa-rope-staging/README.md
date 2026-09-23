# Main-attention RoPE storage — nine GPU tests pass; original request still fails

Relative to the expert-storage control, this diagnostic uses FP32 products
and addition for the main Q/K rotary transform, then stores BF16 once. The
active pinned reference fused_qk_norm_rope.py explicitly normalizes to BF16,
converts back to FP32, loads BF16 cos/sin as FP32, and stores only after the
rotation. DGPP currently rounds each product before adding.

Only the main Q/K calls enable the new policy. Indexer query and pooled-key
rotation retain their existing policy. No frequencies, theta, scaling, cache
indices, query selection, or context limit change.

The new test uses exact normalization and an independent FP64 rotation oracle
at positions including 258300, 261120 and 262143. It requires the legacy policy
to disagree on the same fixture. All QSA tests run before the unchanged TP2
request. Earlier diagnostic precision changes are retained; a successful replay
would require a minimal ablation before promotion.

Independent replay of the 24 captured main-Q final rows reproduces the old
policy exactly, with zero relative L2 error against the actual captured output.
The reference rotation policy changes 5601/18432 rotary values on these inputs.
This verifies a real operator difference, not the cause of retrieval failure.

All nine QSA GPU tests pass, including legacy indexer, compression, long
paged-pool selection, attention and graph checks. The new exact-normalization
rotation test passes all values against FP64; its old-policy control differs.

The unchanged original request still returns the original 5/6 answer, with
184 completion tokens. Bounded prefill was skipped. Production binaries,
resolved configuration and inference were independently verified after restoration.
