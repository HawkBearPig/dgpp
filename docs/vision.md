# Image inputs

GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8-27B and MiniMax-M3.1-Flash accept images through
`POST /v1/chat/completions`. The engine loads the checkpoint's BF16 vision
encoder and projects image features into the language model's prompt. It
supports the FP8 and hybrid NVFP4/FP8 checkpoints, streaming, multiple choices
and speculative drafting (MTP or the DFlash2 block drafter). Other model frontends reject image inputs until they have their own
encoder integration.

Check `GET /v1/models`: a capable model reports
`"input_modalities": ["text", "image"]`. Loading a compatible checkpoint with
`vision_config` enables the encoder automatically; no Python runtime or
external image service is required.

## Request format

Put PNG, JPEG or WebP base64 data URIs in a user message's content array. Text and
images retain their order, including across conversation turns.

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

endpoint = "http://127.0.0.1:18080"
model = json.load(urlopen(endpoint + "/v1/models"))["data"][0]["id"]
image = base64.b64encode(Path("photo.jpg").read_bytes()).decode("ascii")
body = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + image,
            "detail": "auto",
        }},
    ]}],
    "max_completion_tokens": 512,
}
request = Request(endpoint + "/v1/chat/completions",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
print(json.load(urlopen(request))["choices"][0]["message"])
```

## Limits and behavior

| Input | Supported behavior |
| --- | --- |
| Source | `data:image/png;base64,...`, `data:image/jpeg;base64,...` or `data:image/webp;base64,...`; remote URLs, local paths and file IDs are rejected |
| Size | Up to 20 MiB of decoded PNG/JPEG/WebP file bytes, 32 megapixels and 16,384 pixels per source dimension |
| History | Image tokens use the ordinary context budget; no separate image-count or aggregate visual-token cap |
| Decoded data | Up to 256 MiB of resized RGB pixels per request, independently of compressed upload size |
| Detail | `auto` and `high` allow up to 1,024 visual tokens per image (576 for MiniMax); `low` allows 256 |
| Preprocessing | RGB conversion and an aspect-preserving antialiased bicubic resize onto the family's visual-token grid, then the checkpoint's own normalization — see [per-family preprocessing](#per-family-preprocessing) |
| Usage | Each merged patch block contributes one prompt token; image delimiters also count. Normal context and admission limits still apply |
| Prefix cache | Identical image content, geometry and token positions can reuse prompt and generated-continuation snapshots; `cached_tokens` reports reuse |
| Scheduling | Graph prefill can yield between bounded chunks for both image and text requests; active decodes run after each chunk |

## Per-family preprocessing

The served towers disagree about geometry, so the resize policy and the
delimiters are per-family data (`serve/image_inputs.hpp`,
`models/<family>/vision_config.hpp`), not constants in the shared path.

| | GLM-5.3-Flash | Qwen3.8-Flash-Next | Qwen3.8-27B |
| --- | --- | --- | --- |
| Visual token | 28 pixels on a side (patch 14 × merge 2) | 32 pixels on a side (patch 16 × merge 2) | 32 pixels on a side (patch 16 × merge 2), same processor |
| Canvas | rounded **up** to the grid, the remainder painted black | rounded to the **nearest** multiple of the grid (ties to even), never padded | as Flash-Next (shared frontend) |
| Pixel budget | at least 16 tokens, at most 1,024 | at least 64 tokens, at most 1,024 (`min_pixels`/`max_pixels` growth and shrink, so a small image is scaled up rather than refused) | as Flash-Next |
| Normalization | CLIP mean/std, temporal patch duplication | mean = std = 0.5, both temporal frames identical | as Flash-Next |
| Delimiters | `image_start_token_id` 154830, `image_token_id` 154854, `image_end_token_id` 154831 | `vision_start_token_id`, `image_token_id`, `vision_end_token_id` (248053, 248056, 248054), rendered from the tokenizer's own vocabulary | same ids (248053, 248056, 248054) |
| Tower | 24 blocks, RMSNorm, SwiGLU, windowed attention, deepstack injections | 27 blocks, LayerNorm ε 1e-6, GELU MLP, full attention, 2-D axial RoPE with θ 10000, a learned 48×48 position table resampled bilinearly, `deepstack_visual_indexes: []` | same 27-block tower, merger to 5120 (the dense model's width) |
| Where the rows enter | the embedding and the deepstack layers | the embedding only (no deepstack) | the embedding only (no deepstack) |

MiniMax-M3.1-Flash uses a 28-pixel grid, nearest-grid resize, a minimum
four-token pixel budget and a maximum 576-token budget (672² pixels), as
the pinned fast processor does. It uses CLIP normalization and duplicates
the temporal patch of a still image. Its 32-block tower has LayerNorm,
exact GELU, partial 3-D RoPE (78 of each head's 80 dimensions), full
attention, no final LayerNorm, and a projector before the 2×2 merger.
Image delimiters are `]<]start of image[>[`, `]<]image[>[` and
`]<]end of image[>[` (200029, 200025 and 200030). Projected rows enter
the target embedding; DSpark receives the resulting target features.
Its language-model positions stay one-dimensional, as in the published
MiniMax decoder.

MiniMax also accepts user content parts of the form
`{"type":"video_url","video_url":{"url":"data:video/mp4;base64,...","fps":1}}`.
WebM data URIs work too. Native FFmpeg decoding requires a build with
`DGPP_ENABLE_VIDEO=ON` (the default); `/v1/models` advertises `video` only
when both the frontend and model support it. Other model families refuse it.
The API samples uniformly across the complete clip, using 1 fps by default
and an optional positive rate up to 30. It selects 4–768 frames, bounded by
the source frame count, and repeats the last frame when the count is odd.
Per-frame resolution uses the published 28-pixel grid and 4–768-token
budget, further bounded by the processor's 57,600-token total pixel budget
over selected raw frames. Encoded inputs are limited to 32 MiB and decoded
media across a request to 256 MiB. Remote URLs are not fetched.

Every two frames form a temporal patch. A segment contains at most four
temporal patches (eight raw frames), with bidirectional attention inside
that segment and temporal RoPE positions starting at zero for each segment.
Each pair contributes a timestamp `]<]0.0 seconds[>[`, image start/end
delimiters, and one `]<]video[>[` token (200026) per merged spatial patch.
The admission journal carries all RGB frames and the discontiguous token
spans. Prefix-cache identity includes the whole segment even when a cached
cut lies inside its first span: later frames can change earlier embeddings.

Qwen3.8-Flash-Next deviations, both recorded here rather than hidden in a
comment: images are single frames (`video_token_id` is refused, never
decoded), and the tower's rows are spliced into a prompt whose positions are
still one-dimensional. The same holds for Qwen3.8-27B (shared frontend and
position design), with one of its own: the DFlash2 draft embeds the pad ids
directly, so drafts at image rows are weaker while the verify stays exact;
measured on image-heavy traffic the damage is negligible (p1 acceptance
74.5%, inside the 70-89% text-only band; 36.4% of draft tokens accepted
overall). Prefix reuse across image requests is measured too: a repeated
image prompt attaches 1,722 of 1,727 cached tokens, first sight caches
nothing, and a TTFT-hit request serves in ~133 ms against ~470 ms misses. The checkpoint's language model expects interleaved
mRoPE, where an image span advances the height and width axes independently;
DGPP's position array is also its cache-slot index, so the three-axis
positions land with that split, not before it. Spatially demanding prompts
(fine-grained layout, dense tables, OCR of long rows) are the ones to
re-measure when it does. Until then the served behaviour is a usable but
degraded reading of the reference.

The GLM processor targets at least 16 visual tokens for small images. Aspect
ratio and grid alignment determine the actual count. DGPP's 1,024-token
per-image ceiling bounds the native encoder's workspace; it is lower than
the upstream processor's 8,000-token default. Images are decoded and resized
on the HTTP head before admission, so preparation of a large image can delay
other HTTP work. PNG/JPEG/WebP decoding does not apply EXIF orientation or color
profiles. WebP supports lossy and lossless still images; alpha is composited
on white before resizing, and animated WebP is unsupported. PDF file inputs
still extract text only; page rasterization is not part of this feature.
Video and audio inputs are unsupported.

Images of the same dimensions have identical placeholder token IDs but
different embeddings. Cache lookups compare processed RGB pixels, geometry
and token positions as well as the text prefix; a hash match alone is not
enough. Repeated turns share immutable image data in the index. A changed or
new image can still reuse an earlier prefix, before that image affects model
state. Suffix prefill encodes only images that are not wholly covered by the
attached snapshot, including images straddling the attach position. The
cache identity includes MTP's next-token image dependency at a snapshot cut.
The prefix index shares immutable pixels and retains at most 256 MiB of unique
pixel identities. When this budget is full, new cache insertions are skipped;
requests still execute. Metrics expose `image_bytes` and `skipped_image_bytes`.

The encoder reuses one image-output buffer and a 2,049-row staging window.
Qwen splits wider model chunks into windows, copying each window before
staging the next. Each resumable cursor borrows its request's image vector;
the model exposes that borrow only during a prefill call and invalidates the
shared window on entry and exit. Main-model and MTP consumers finish
before the window is reused. An image crossing a chunk or attachment boundary
is encoded in full, then only the required rows are staged. The single-image
output can serve successive chunks without re-encoding. Historical images
already represented by an attached prefix need no encoder work.

## Deployment and validation

Each rank loads approximately 1.05 GiB of replicated vision weights and
reserves 0.33 GiB of workspace. The startup memory plan includes both, even
for text-only traffic. Workspace is allocated before collective execution
starts; image requests do not allocate CUDA buffers while another rank may
be spinning in a collective. Resized RGB bytes and token offsets travel in
the admission journal so every rank uses the same input.

MiniMax instead reserves 1.615 GiB of visual weights and 1.789 GiB of workspace
per rank. This includes a 384 MiB cache for eight encoded images or eight-frame
segments, so interleaved readers can reuse complete embeddings across prompt
chunks. Cache entries compare geometry and every frame's pixels; prompt offsets
only control embedding placement. Least-recently-used entries are evicted and
recomputed without changing the numerical recipe. The retained RGB identities
use at most another 147 MiB of host memory per rank. It stages up to 513 rows
per window. The encoder uses
FP32 SGEMM for the patch projection, FP32 scores with BF16 probabilities,
and BF16 projections with the validated 1 MiB cuBLASLt workspace recipe.
The FP32 rotary tables are built on CUDA. These choices preserve the
reference's rounding across the full 32-layer tower.
For segments above 9,216 patch rows, attention softmax follows the reference's
four-value reduction order, including its shared-memory cutoff and scalar
tail. The shorter still-image reduction cannot be extended unchanged to
16,384 patch rows without accumulating full-depth error.

For the MiniMax encoder, build `minimax_vision_check` and run the real-weight
regressions on an idle GPU (the oracle requires NumPy and PyTorch):

```bash
python3 tools/minimax_vision_regression.py --checkpoint-dir "$CKPT" \
  --binary build-release/minimax_vision_check --output-dir build-minimax-vision-regression
```

Eight fixed image cases cover gradients, noise, checkerboards, rectangular images,
the 128-query tile boundary, the processor's 576-token budget and the
encoder's 1,024-token allocation limit. All eight final embeddings match
the CUDA reference bitwise with the pinned real checkpoint. Eight additional
video cases cover 2/4/6/8-frame segments, temporal coordinates and the
largest allocated segment. `--media image` or `--media video` selects a
subset for focused development checks. The numerical
gate remains relative RMS ≤0.005 and cosine ≥0.99998. See the
[MiniMax implementation record](minimax_m31_flash_plan.md#native-still-image-encoder)
for checkpoint and arithmetic details; encoder parity alone does not certify
end-to-end visual task accuracy.

The lossless API fixtures `tests/data/minimax_video_10frames.mp4` and `.webm`
contain ten 56×56 constant RGB frames at 2 fps. Frame `i` is
`[17*i, 255-11*i, 10+7*i]`. They were encoded with FFmpeg 6.1.1 using
`libx264rgb -crf 0 -preset veryslow -pix_fmt rgb24 -movflags +faststart` and
`libvpx-vp9 -lossless 1 -pix_fmt gbrp`, respectively. The host regression
checks exact decoded pixels, frame selection, odd-frame repetition,
segment boundaries, timestamp text, token spans, and invalid inputs.

On an idle test deployment, run:

```bash
python3 scripts/vision_api_check.py --help
python3 scripts/vision_api_check.py --url http://127.0.0.1:18080
python3 scripts/vision_prefix_cache_check.py --url http://127.0.0.1:18080
# MiniMax needs more images to cross the same 12,288-token history gate:
python3 scripts/vision_prefix_cache_check.py --url http://127.0.0.1:18080 --history-images 24
python3 scripts/prefill_fairness_check.py --url http://127.0.0.1:18080
python3 scripts/prefill_fairness_check.py --url http://127.0.0.1:18080 --images
python3 scripts/prefill_fairness_check.py --url http://127.0.0.1:18080 --images --image-count 12 --decoders 2
```

The check generates its own images and exercises different image content,
multiple images, streaming, multiple choices, large images, prefill chunk
boundaries and concurrent requests. Stop the deployment and compare rank
operation-stream hashes as described in [operations](operations.md).

For encoder arithmetic, build `glm_vision_check` and set `CKPT` to the
GLM-5.3-Flash snapshot directory. The optional oracle requires NumPy and
PyTorch; serving does not. `qwen_vision_check` is the same harness for
Qwen3.8-Flash-Next, with `tools/qwen_vision_reference.py` as its oracle and a
32-pixel canvas (`64 64` is 2×2 visual tokens).

`glm_vision_stream_test` compares staged rows bitwise with whole-image
encoder outputs across 12 full-size images, 256/2,048-token windows, MTP
lookahead and an attachment inside an image. It uses the cached serving
checkpoint, or `DGPP_VISION_TEST_CHECKPOINT` when set. Run this GPU test
with serving stopped.

`qwen_image_prefill_test` uses synthetic text and vision weights to check
cold image prefills, interleaved cursors with and without MTP, cancellation,
slot reuse, and MTP catch-up at an attached prefix. It needs a GPU but no
downloaded checkpoint.

The 27B model has corresponding `qwen35_image_prefill_test` and
`qwen35_mixed_image_prefill_test` gates for the FP8 and mixed NVFP4 formats.
They also check grouped image requests and the shared prefill task's ownership
of prompts, pixels and snapshot boundaries across yields.

```bash
cmake --build build-ci -j 4 --target glm_vision_check
build-ci/glm_vision_check "$CKPT" 112 112 /tmp/vision.bf16 /tmp/vision-trace
python3 tools/glm_vision_reference.py "$CKPT" 112 112 /tmp/vision.bf16 \
  --device cuda --trace-dir /tmp/vision-trace --isolate-layers
python3 tools/glm_vision_reference.py "$CKPT" 112 112 /tmp/vision.bf16 --device cuda
```

The GLM oracle targets CUDA BF16 eager attention and enforces a full-depth
relative RMS bound of 0.5% and cosine similarity of at least 0.99998. The same
bounds apply to isolated blocks and operations; patches must match exactly.
The pinned 30-case regression runner requires bitwise matching final embeddings.
Use `--isolate-ops` with traces to locate the first mismatch. The checker
accepts `-` as its trace directory to suppress dumps, and an optional final
regular expression limits the traced stages. `--diagnostic` on the oracle
reports distances without declaring parity; `--attention fp32` preserves the
initial investigation's different attention contract.

The numerical investigation
records the fixes and comparisons against the same eager reference. All
five synthetic cases and 25 pinned diagram cases have bitwise identical final
embeddings. The diagrams contain 18 distinct images: this is a small
regression sample, not a guarantee of general vision quality or identical
outputs across attention backends. The
initial validation also records
input plumbing and controlled answer comparisons.

GLM encoder equations and preprocessing follow the official Transformers
[GLM5-Next implementation](https://github.com/huggingface/transformers/blob/f0d778337771dd81082653d752f8bd6563b1d2eb/src/transformers/models/glm5_next/modeling_glm5_next.py).
All projections explicitly select FP32 reductions followed by BF16 output,
with a fused epilogue when a bias is present. Final LayerNorm follows the
CUDA reference’s Welford reduction order before BF16 rounding and GELU. Vision
RMSNorm uses FP32 reductions and separate BF16 normalization/weight steps.
Attention rounds QK scores and scaling to BF16 before FP32 softmax, then
rounds probabilities to BF16. It batches heads and tiles queries while using
the untiled matrix's algorithm, preserving the reduction order without
increasing workspace. The oracle disables TF32 and reduced-precision BF16
reductions. Serving does not depend on PyTorch.

For Qwen, build `qwen_vision_check` and `qwen_vision_test`, and set `CKPT` to
the Qwen3.8-Flash-Next snapshot directory:

```bash
cmake --build build-ci -j 4 --target qwen_vision_check qwen_vision_test
ctest --test-dir build-ci --output-on-failure -R '^qwen_vision_test$'
build-ci/qwen_vision_check "$CKPT" 1024 1024 /tmp/qwen-vision.bf16
python3 tools/qwen_vision_reference.py "$CKPT" 1024 1024 /tmp/qwen-vision.bf16
```

The Qwen oracle checks the native encoder's BF16 projections and probabilities
with FP32 attention scores and scaling. Blocks use tanh GELU; the merger uses
exact GELU. The same 0.5% relative RMS and 0.99998 cosine limits apply, without
asserting bitwise parity with Transformers attention backends that round scores
to BF16. The PR follow-up validation
records bitwise agreement for four deterministic canvases, including the
1,024-token ceiling, plus memory and race sanitizer checks. These encoder
checks do not cover the language-model mRoPE limitation described above.
