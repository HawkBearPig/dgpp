# First-layer attention history reconstructed from tokens — completed

This CPU audit removes the captured-input assumption from the first layer's
attention calculation. It starts with the retained 261290-token forced-prefix
fixture and checkpoint tensors, independently builds the input hyper-state
mixer, dense projections, complete convolution and recurrent history from zero,
gated normalization, output projection and TP2 fold. All 48 value heads are
replayed, rather than the two sampled first-layer heads in the earlier history
audit. Captured activations and state are used only for comparison.

The fixture has 354 distinct token IDs. Before the first recurrent layer,
embedding and mixer calculations depend only on each token, so independent
FP64 products can be evaluated once per distinct ID and expanded by the
original sequence. The native BF16 storage and block-FP8 weight conversion
policies are retained. The a/b projection weights remain BF16 as in the loader;
they are not converted to FP8. All checkpoint convolution, decay, bias and norm
constants match their actual captured values bitwise on both ranks.

`check_recur.py` first compares the standalone C++ recurrence with a separate
vectorized NumPy implementation on 1201 actual token operands, all 48 heads,
and a state checkpoint at position 7. All BF16 core/convolution/state-history
outputs match exactly; FP64 state values agree within 1e-12 absolute/relative
tolerance. This checks head grouping, state orientation, padding, history
updates and the checkpoint boundary before the long replay is accepted.

## Full-prefix measurements

Native comparisons cover both complete recurrent states at positions 260096
and 261290 and all 1194 final-chunk rows. These are measured differences from
higher-accuracy arithmetic, not claimed bitwise equivalence or a newly chosen
pass tolerance.

| Comparison | Rank 0 relative L2 | Rank 1 relative L2 |
| --- | ---: | ---: |
| Input GR mixer, final chunk | 0.000204533 | 0.000204533 |
| QKV projections, final chunk | 0.000359157 | 0.000355745 |
| State before final chunk | 0.002724077 | 0.001213674 |
| State after final chunk | 0.002200960 | 0.001025797 |
| Convolved QKV, final chunk | 0.000342702 | 0.000403428 |
| Recurrent core, final chunk | 0.002019155 | 0.001683949 |
| Gated norm, final chunk | 0.001631290 | 0.001352361 |

The final-chunk output projection and rank fold have relative L2 **0.002316724**
and maximum absolute error **0.0078125**. All values are finite. The worst
individual head's final-chunk core relative L2 is **0.003446192** (rank 0,
local head 19). Per-row/per-head outliers and counts are retained in
`replay.json`; aggregate agreement does not imply every scalar agrees.
The independent recurrence takes 32.98 seconds with four CPU threads and
33824 KiB peak RSS. Production remains online; no GPU world is launched.

This validates the coverage of the independent reconstruction, not a retrieval
repair. FP64 projection/norm/recurrence arithmetic can differ from native FP32
accumulation; the replay does not identify a violated numerical contract.
Only layer zero's attention path is rebuilt. Its MoE and residual injection,
and the earlier input construction of subsequent layers, remain outside this
record. The original unchanged 261120-token request is a separate fixture;
this audit uses its previously verified 261290-token forced-prefix extension.

The next bounded extension can carry this independent history through the
first layer's residual and expert computation at selected record/outlier
positions, comparing against the retained full-history pre-PLE residuals.
That requires no additional reporter material or full BF16 checkpoint run.

## Reproduction

With the retained checkpoint and operator captures available:

1. Run `prepare.py` using CPU NumPy and `OPENBLAS_NUM_THREADS=2`.
2. Compile `recur.cpp` with C++20, `-O3 -march=native -fopenmp -ffp-contract=off`.
3. Run `check_recur.py`; require `selfcheck.json` to pass.
4. Run `OMP_NUM_THREADS=4 raw/recur raw/input raw/output`.
5. Run `analyze.py` using CPU NumPy and `OPENBLAS_NUM_THREADS=2`.

Scripts refuse to overwrite prepared input or recurrence output directories.
JSON receipts bind the checkpoint tensors, generated operands, actual captures,
executable and source files. Raw inputs, outputs and logs remain local.
