# Final-row residual injection boundaries — independent replay completed

The earlier GR audit covered only the attention input mixer and omitted the
PLE layer. This replay includes all 48 layers at the failed forced-prefix row
(position 261289), both residual injection steps, and the intervening MoE input
mixer. It uses the retained operator and MoE captures. No GPU serving run or
inference source change is needed.

For each layer it starts from the captured preceding residual (the checkpoint
embedding for layer zero, the archived post-PLE residual for layer one).
It independently reconstructs checkpoint block-FP8 down/up matrices, grouped
normalization, projections and staged sigmoid/product arithmetic. FP64 products
and explicit BF16 rounding are used. It carries its own intermediate residual
through attention injection, the MoE mixer and MoE injection, while accepting
the captured folded attention and folded expert branch outputs.

Results on rank-replicated fields:

| Comparison | Elements | Worst relative L2 | Exact elements |
| --- | ---: | ---: | ---: |
| Attention input mixer | 122880 | 0.001821159 | 121474 |
| Attention injection through MoE input mixer | 122880 | 0.000966905 | 122381 |
| Both injections through final residual | 491520 | **0** | **491520** |

Both ranks' captured inputs and outputs match bitwise. All results are finite.
The earlier 94 rank/layer attention-mixer metrics agree within 2.2e-19;
layer one is now included using the actual post-PLE operand. Both complete
5.35 GB post-PLE source files were freshly rehashed against the verified archive
manifest before their 20 KiB final rows were exported. CPU replay took 12.821 s.

This establishes exact combined residual-injection output on the sampled row,
not individual intermediate injection equality where no capture exists. The
operator and MoE captures were collected separately and each previously passed
clean-response/logprob parity. Earlier residuals and branch outputs are inputs
to this replay; it is not an independent whole-model execution, an audit of
all prompt positions, or a retrieval fix. The capture's old EOS policy is
retained intentionally in its actual post-PLE operand; the separate EOS fix
is not reversed or retested here.

`export_ple.py` retrieves the bounded archived operand; `replay.py` runs with
CPU NumPy (`OPENBLAS_NUM_THREADS=2`). `replay.json` includes per-layer errors,
checkpoint-tensor hashes and hashes of all consumed input tails. Raw data is
retained locally and excluded from version control.
