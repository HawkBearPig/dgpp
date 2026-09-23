# Native chunked GDN prototype — not validated

The recurrent-GDN reference control reproduces the original DGPP failure twice;
the fresh unmodified Marlin baseline also fails twice. The GDN swap therefore
does not establish causality. This prototype remains diagnostic.
The native prototype implements the 64-token WY block formulation with BF16
matrix intermediates and FP32 persistent state. Decode and speculative snapshots
keep the original recurrent path. No other diagnostic arithmetic changes are
included; the worktree starts from the separate EOS fix.

A standalone CUDA library builds. The server build passed. The running
comparison checks the pinned reference on 16 frozen real cases, six block-edge
lengths, distinct mixed inputs at the real 8/24 head geometry, and a small deterministic
regression fixture. A separate frozen-input probe varies FLA launch configurations. Predefined
budgets are relative output L2 below 0.002 and final-state L2 below 0.001, finite
outputs and bitwise repeatability. These are operator checks, not an accuracy
fix verdict. The original long request must pass afterward.
