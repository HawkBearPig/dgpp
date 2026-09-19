# GLM image streaming and prefill scheduling

The GLM-5.3-Flash image path now stages embeddings through a fixed
single-image buffer and a 2,049-token window. The window covers one
2,048-token language-model chunk plus MTP lookahead. Visual tokens use the
ordinary context budget; image history has no separate count/token cap.
Decoded request pixels and shared prefix-cache pixel identities each have
a 256 MiB host-byte budget. Exhausting the latter skips cache insertion.

Vision workspace is 355,983,360 bytes (0.332 GiB) per rank, including all
encoder scratch. The image-output and window buffers together use
25,174,016 bytes, independent of history length. Runtime buffer-capacity
checks prevent growth after startup.

GLM graph prefill uses the existing replicated scheduler continuation path.
The four-rank deployment sets `prefill_budget_tokens=256` and
`prefill_idle_budget_tokens=2048`. Every advance completes main-model and MTP
work for its chunk before yielding. Unfinished slots have device positions
masked during padded decode graphs. Cursor epochs reject stale continuations
after cancellation and slot reuse.

## Validation

- Release and CI builds pass; CI enables warnings as errors.
- Image unit tests: 7 passed, including histories above the old limits,
  context bounds, invalid token/pixel geometry and decoded-byte exhaustion.
- Host CTest suites: scheduler, HTTP service, fabric journal and checkpoint
  vision frontend all passed. Coverage includes image prefill/decode
  interleaving, cancellation and host-byte accounting. The final fabric run
  also executed its three registered cases (3.06 seconds including the legacy
  suite): image validation, a 12-image/12,288-visual-token journal round trip,
  and a fragmented 64 MiB journal record followed by empty/short/partial lines.
- `glm_tp_resumable_prefill*`: 2 GPU fixture tests passed. Main and draft
  logits match identical synchronous cuts bitwise. Tests cover interleaved
  peer decode, mid-prefill snapshots, close-snapshot MTP catch-up, cancellation,
  stale cursors and complete reservation release.
- `glm_vision_stream_test`: staged image rows match independent whole-image
  encoder outputs bitwise using the actual serving checkpoint. Twelve
  1,024-token images are checked at 256- and 2,048-token windows, including
  one-token lookahead and a resume inside an image.

GPU checks ran with the serving deployment stopped. The prior deployment
was idle before shutdown and all four operation streams matched.

Compute Sanitizer memcheck also passed both resumable-prefill GPU tests,
with zero reported errors.

## Image history and admission

All 15 live image/cache cases passed on the final four-rank release build:
cold/warm/cache-disabled prompts, generated continuations, changed pixels,
new images after attachment, images crossing chunk boundaries, twelve
full-size images, twelve mixed small images, and concurrent red/blue requests.
A 6,071-token image prompt took 4.710 seconds cold and 0.127 seconds warm,
reusing 6,068 tokens. The twelve-full-size-image history took 12.647 seconds
cold and 0.499 seconds warm, reusing 12,448 of 12,451 prompt tokens.

The large-history test exposed a separate journal bottleneck: after each
4 KiB socket read, the receiver searched the entire growing record for its
newline. Incremental scanning removes that quadratic work. Before this fix,
the corresponding cached twelve-image request took 11.645 seconds despite
only about 0.605 seconds of model prefill. The final 0.499-second result
includes admission, prefill and completion. The fragmented 64 MiB host
regression covers this path and preservation of subsequent buffered lines.

Command: `python3 scripts/vision_prefix_cache_check.py --url http://127.0.0.1:18080`.

## Live scheduling

The actual four-rank release deployment processed a cold 21,043-token text
prompt while another request streamed a 1,024-token completion. The active
stream emitted 154 content/reasoning events during 23.514 seconds of observed
prefill. Its largest event gap was 0.310 seconds, including admission. The
new request completed with the expected answer and zero cached prompt tokens.
`engine_failed` remained false.

Command: `python3 scripts/prefill_fairness_check.py --url http://127.0.0.1:18080`.
The script measures gaps across admission as well as during reported prefill;
it asserts progress before prefill completion, rather than relying only on
final request success.

The same test with two full-size images and 23,096 prompt tokens emitted
171 decode events during 26.675 seconds of prefill. The maximum event gap,
including admission, was 0.802 seconds.

A third test kept two decode streams in slots 1 and 2, then assigned the
image prefill to the freed slot 0. This exercises an unfinished slot inside
the padded batched graph, rather than only scalar decoding. Each stream
emitted 180 events during 28.297 seconds of prefill (23,093 tokens); the
largest event gap was 0.774 seconds. All requests completed, the new request
answered correctly, and no engine or attention-anomaly errors were logged.

Commands add `--images`, then `--images --decoders 2` to the scheduling check.
The two-decoder mode automatically creates and retires a short slot-0
request before submitting the measured prefill.

The final build, including incremental journal scanning, also passed
`--images --image-count 12 --decoders 2`: a cold 33,353-token prompt with twelve
full-size alternating-color images occupied slot 0 while slots 1 and 2
decoded. Each stream emitted 261 events during 48.227 seconds of observed
prefill; the maximum event gap, including admission, was 0.820 seconds.
The prefill answered correctly with zero cached tokens and both decodes
completed their 1,024-token budgets.

These are observed gaps on the idle four-node test deployment, not latency
bounds for arbitrary prompts, image sizes or load.

After the final image and scheduling runs, the deployment had no active,
queued or prefilling requests, zero failed requests, and `engine_failed=false`.
All four operation streams had SHA-256
`c2de82be392370d2b12c48f415e94962f7d0814036d06de08c74c3b4630aabdc`.
The running head binary matched the release build and remains deployed.

## Large mixed-image answer diagnostic

A 12-turn history containing eleven full-size red images followed by one
full-size blue image produced the answer `Red` when asked for the final
image's color. Both the streaming build and an isolated control that encodes
and retains all images upfront produced the same answer (12,416 prompt
tokens, cache disabled). The control used a startup-allocated aggregate
embedding buffer and the same model, MTP and token cuts; it was built under
`/tmp` and is not part of the implementation. Two full-size images and twelve
small images answered `Blue` correctly. This rules out streaming as the
cause of that particular answer error; it does not establish general
multi-image answer accuracy.

The live limit regression therefore checks twelve full-size images of one
color, while separate mixed-image cases and exact encoder comparisons check
pixel identity and ordering. The temporary control was removed from serving
after the comparison.
