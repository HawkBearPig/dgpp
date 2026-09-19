# Image inputs

GLM-5.3-Flash accepts images through `POST /v1/chat/completions`. The engine
loads the checkpoint's BF16 vision encoder and projects image features into
the language model's prompt. It supports the FP8 and hybrid NVFP4/FP8
checkpoints, streaming, multiple choices and MTP. Other model frontends reject
image inputs until they have their own encoder integration.

Check `GET /v1/models`: a capable model reports
`"input_modalities": ["text", "image"]`. Loading a compatible checkpoint with
`vision_config` enables the encoder automatically; no Python runtime or
external image service is required.

## Request format

Put PNG or JPEG base64 data URIs in a user message's content array. Text and
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
| Source | `data:image/png;base64,...` or `data:image/jpeg;base64,...`; remote URLs, local paths and file IDs are rejected |
| Size | Up to 20 MiB of decoded PNG/JPEG file bytes, 32 megapixels and 16,384 pixels per source dimension |
| History | Image tokens use the ordinary context budget; no separate image-count or aggregate visual-token cap |
| Decoded data | Up to 256 MiB of resized RGB pixels per request, independently of compressed upload size |
| Detail | `auto` and `high` allow up to 1,024 visual tokens per image; `low` allows 256 |
| Preprocessing | RGB conversion, aspect-preserving antialiased bicubic resize, black right/bottom padding to a 28-pixel grid, CLIP normalization and temporal patch duplication |
| Usage | Each merged 28×28 patch contributes one prompt token; image delimiters also count. Normal context and admission limits still apply |
| Prefix cache | Identical image content, geometry and token positions can reuse prompt and generated-continuation snapshots; `cached_tokens` reports reuse |
| Scheduling | GLM graph prefill can yield between bounded chunks for both image and text requests; active decodes run after each chunk |

The processor targets at least 16 visual tokens for small images. Aspect
ratio and grid alignment determine the actual count. DGPP's 1,024-token
per-image ceiling bounds the native encoder's workspace; it is lower than
the upstream processor's 8,000-token default. Images are decoded and resized
on the HTTP head before admission, so preparation of a large image can delay
other HTTP work. PNG/JPEG decoding does not apply EXIF orientation or color
profiles. PDF file inputs still extract text only; page rasterization is not
part of this feature. Video and audio inputs are unsupported.

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

The encoder reuses one image-output buffer and a 2,049-row staging window
(one prefill chunk plus MTP lookahead). Main-model and MTP consumers finish
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

On an idle test deployment, run:

```bash
python3 scripts/vision_api_check.py --help
python3 scripts/vision_api_check.py --url http://127.0.0.1:18080
python3 scripts/vision_prefix_cache_check.py --url http://127.0.0.1:18080
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
PyTorch; serving does not.

`glm_vision_stream_test` compares staged rows bitwise with whole-image
encoder outputs across 12 full-size images, 256/2,048-token windows, MTP
lookahead and an attachment inside an image. It uses the cached serving
checkpoint, or `DGPP_VISION_TEST_CHECKPOINT` when set. Run this GPU test
with serving stopped.

```bash
cmake --build build-ci -j 4 --target glm_vision_check
build-ci/glm_vision_check "$CKPT" 112 112 /tmp/vision.bf16 /tmp/vision-trace
python3 tools/glm_vision_reference.py "$CKPT" 112 112 /tmp/vision.bf16 \
  --device cuda --trace-dir /tmp/vision-trace --isolate-layers
python3 tools/glm_vision_reference.py "$CKPT" 112 112 /tmp/vision.bf16 --device cuda
```

The default oracle targets CUDA BF16 eager attention and enforces a full-depth
relative RMS bound of 0.5% and cosine similarity of at least 0.99998. The same
bounds apply to isolated blocks and operations; patches must match exactly.
The pinned 30-case regression runner requires bitwise matching final embeddings.
Use `--isolate-ops` with traces to locate the first mismatch. The checker
accepts `-` as its trace directory to suppress dumps, and an optional final
regular expression limits the traced stages. `--diagnostic` on the oracle
reports distances without declaring parity; `--attention fp32` preserves the
initial investigation's different attention contract.

The [numerical investigation](../benchmarks/results/2026-09-18-glm-vision-numerics.md)
records the fixes and comparisons against the same eager reference. All
five synthetic cases and 25 pinned diagram cases have bitwise identical final
embeddings. The diagrams contain 18 distinct images: this is a small
regression sample, not a guarantee of general vision quality or identical
outputs across attention backends. The
[initial validation](../benchmarks/results/2026-09-18-glm-vision.md) also records
input plumbing and controlled answer comparisons.

Encoder equations and preprocessing follow the official Transformers
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
