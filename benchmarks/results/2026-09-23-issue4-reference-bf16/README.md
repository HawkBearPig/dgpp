# Full BF16 reference feasibility — metadata only

The official `Qwen/Qwen3.8-Flash-Next` snapshot
`de4b8e4d43b917e7706784d8bb445c9af86a3540` is available and ungated.
Its text configuration exactly matches the tested FP8 snapshot. It has no
quantization config and contains 359999963128 tensor bytes in 131 weight files
(360000192888 bytes including headers; largest file 3510240000 bytes).
Only public metadata, configuration, index and model card were read.
**No BF16 weight download or inference has started.**

A full BF16 control could test whether the original association also fails
without quantized expert weights and n-gram tables. It would require a separate
four-GPU run and substantial model storage. It changes the weight representation,
expert backend and tensor-parallel decomposition, so success alone would not
isolate a DGPP operator defect. It is optional, not a prerequisite for continuing
the engine investigation or implementing a demonstrated fix.

Source: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540
