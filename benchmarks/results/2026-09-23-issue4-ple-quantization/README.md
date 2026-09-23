# PLE precision isolation — GPU validation passed, request comparison running

The independent FP8 reference still shares its quantized n-gram table with the
NVFP4 checkpoint. This experiment isolates that remaining shared quantization:
retain the original NVFP4 experts, dense projections, native TP2 arithmetic,
configuration and unchanged request, replacing only table lookup values with
the original BF16 rows. This is a diagnostic hypothesis, not a retrieval fix.

## Completed CPU checks

`check_table_samples.py` reads bounded byte ranges from the pinned public
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`
checkpoint and compares them with the resident FP8 checkpoint. The 42 requests
read 601392 bytes, including headers and shared tensor samples. Every response
must honor the requested range; the script refuses a full-file response.

The 1536 sampled table rows contain 245760 values across eight shards. Their
FP8 dequantization has relative L2 error 0.0266871 against BF16. Requantizing with
the stored global scale reproduces 240210 codes exactly; all other magnitudes
are within one FP8 step, with no nonzero sign disagreements or nonfinite codes.
The sampled shared projection weights and complete hash/norm buffers all match.
This finds no large scaling/conversion discrepancy in those samples. It neither
audits every table row nor establishes whether this quantization affects retrieval.

## Diagnostic implementation and pending validation

`diagnostic.patch` applies to EOS-corrected base
`d318815984fd2ab1f0a67a6b830b57730a57439a`, which already reproduces the original
5/6 failure. The default path retains FP8 table lookup. With
`DGPP_ISSUE4_PLE_BF16_MANIFEST`, the loader validates table geometry/hash buffers,
maps the rank's BF16 rows and copies their bits into the existing embedding
buffer. Key/value projections and all later computation stay on their existing
paths. Each rank needs only files intersecting its table row range.

The release server and focused test executables build. A new CPU regression
passes exact BF16 row/gather checks, including serial and parallel gathering,
cross-shard rows, head selection, absent nonlocal shards and rank bounds.
A GPU integration control also passes: store the original FP8-derived values
as BF16 and compare eager/MTP graph transcripts on a synthetic TP2 fixture.
All 11 focused checks pass (three loader tests, one storage integration control
and seven PLE kernel tests). The unchanged-request A/B/A campaign is running:
original FP8 tables, original BF16 tables, then original FP8 tables again.
No retrieval verdict is available yet.

The first diagnostic attempt failed during CUDA graph capture with the new
host-to-device memcpy path. Its FP8 control completed, but the BF16 storage
control failed before any original request ran. Production restoration passed.
`attempt1.json` excludes this attempt from accuracy evidence. The corrected
diagnostic uses the existing `copy_rows_bf16` kernel to read mapped host staging;
the unchanged storage control now passes through eager and MTP graph decoding.

The 33 BF16 snapshot files containing PLE tables total 104298732704 bytes.
`table-file-manifest.json` pins their published size/SHA256; the downloader
verifies every complete file before use. All 33 files on node 12 and all 17
files needed by rank 0 on node 11 pass full size/SHA256 checks. This is a subset
of the full 360 GB checkpoint, and no full BF16 model run is planned for this comparison.
Node 12's direct download failed at DNS before receiving data; a temporary
loopback SOCKS proxy through the head node completed the download and has been closed.

## Retained capture relocation

The previous full-history captures occupy 80845836686 bytes. They were copied
to node 13 at `/home/stephen/dgpp-issue4-archive-20260923/history-captures`.
Independent SHA256 verification matches all 1158 files and the exact file set.
Only after that verification was the old local copy removed to make room for
rank 0's BF16 table files. Both manifests and `capture-relocation.json` preserve
the location and checksums. Production ran during CPU/storage preparation. It
was stopped for the GPU checks and request comparison; the campaign restores
the original four-node binary/configuration and checks inference in `finally`.
