# Dependencies

Start with the [README](../README.md) for the setup commands and dependency
guidance. Source builds link to the detailed getting-started guide; packaged
releases list their runtime dependencies in their own README.

PNG/JPEG image inputs use the vendored `stb_image.h`, compiled directly into
the service. No additional runtime image library is needed. Its pinned source
and license are recorded in [licenses/stb_image.md](licenses/stb_image.md).
WebP uses the vendored `simplewebp.h`, also compiled into the service. Its
source and local fixes are recorded in `third_party/simplewebp/README.md`; the
BSD license and WebM patent grant ship in
[licenses/simplewebp.md](licenses/simplewebp.md).

Native MP4/WebM video input links FFmpeg libraries when
`DGPP_ENABLE_VIDEO=ON` (the default). Source builds need `libavformat-dev`,
`libavcodec-dev`, `libavutil-dev` and `libswscale-dev`; Ubuntu 24.04 serving
peers need the corresponding FFmpeg 6 runtime libraries. No FFmpeg CLI
process is used for requests. `DGPP_ENABLE_VIDEO=OFF` keeps still-image
support and rejects video inputs explicitly. See [build setup](getting-started.md).

The vision LayerNorm reduction schedule adapts PyTorch CUDA arithmetic under
the [PyTorch license](licenses/pytorch.md). PyTorch is only needed for optional
reference checks; it is not a serving dependency.

JSON Schema patterns/formats and custom-tool grammars use PCRE2 10.45.
CMake prefers system PCRE2 development files; without them it downloads a
hash-pinned source archive and links PCRE2 statically. Offline builds can
provide `FETCHCONTENT_SOURCE_DIR_PCRE2`. Its license ships in
[licenses/PCRE2.md](licenses/PCRE2.md). When linking a system shared library,
that library must also be installed on serving peers.

PDF inputs require `pdftotext` (`poppler-utils`) on the HTTP head only. The
configured executable is spawned directly, without a shell. Text/code file
inputs do not require Poppler. Workers, extraction timeouts, decoded byte
limits and uploaded-file storage are configured in `engine.file_inputs`.
