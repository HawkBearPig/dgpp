# OpenAI API compatibility and live metrics

Audited against the official OpenAI reference on 2026-09-18:
[Chat Completions creation](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [Chat resources](https://developers.openai.com/api/reference/resources/chat).
The document/tool extensions follow the official guides for
[file inputs](https://developers.openai.com/api/docs/guides/file-inputs),
[custom tools](https://developers.openai.com/api/docs/guides/function-calling),
and [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

DGPP implements text, document and GLM-5.3-Flash image inputs, function tools
and custom tools through Chat Completions.
It is not an implementation of every OpenAI platform feature. Features depend
on the checkpoint, tokenizer and engine; `/v1/models` reports DGPP capability
extensions. Model sampling defaults come from the checkpoint configuration.

## Chat Completions

| Surface | DGPP behavior |
| --- | --- |
| `POST /v1/chat/completions` | One-shot JSON or SSE with stable completion ID, model, timestamp, indexed choices, finish reason and usage. |
| Messages | Developer, system, user, assistant and tool roles; text strings, text parts and user file/image parts when supported; assistant function/custom-tool history and refusal text. Developer instructions map to the checkpoint's system role. Standard tool results require `tool_call_id`; text parts are joined in order. |
| Limits | `max_completion_tokens` or legacy `max_tokens`, including generated reasoning tokens; send one. Positive integers up to the implementation's 32-bit bound, further limited by KV capacity. `n` accepts 1–128, subject to admission capacity. `/v1/models` reports the resulting number: `context.request_limit_tokens` (the lesser of the model's positional ceiling and the K/V pool), with `position_ceiling_tokens`, `kv_pool_tokens` and `limited_by` beside it, and `rope_scaling` while an `engine.rope_scaling` ramp is in force. |
| Sampling | Temperature, top-p, presence/frequency penalties, signed 64-bit seed, logit bias, and logprobs when supported by the engine. `top_p: 0` selects the highest-probability candidate. Nullable standard options treat null as omitted. |
| Stops | One to four nonempty stop strings, removed from visible content. Usage still counts the generated tokens that completed the match. |
| Function tools | Up to 128 unique functions; auto, none, required, named function and `allowed_tools` subset selection. Required/named/subset selection, strict schemas and disabling parallel calls require constrained decoding. Without constraints, auto is prompt-driven and none omits tools from the prompt. |
| Structured output | Text, JSON object, or the supported JSON Schema subset; constrained decoding required for JSON. Local `$ref`/`$defs` (including recursive schemas), string patterns/formats and exact decimal `multipleOf` are enforced. Strict unsupported schemas are rejected with the keyword path. Non-strict schemas can fall back to unconstrained JSON, with a server log warning. See the grammar implementation/tests for the supported subset. |
| Logprobs | Content token entries with token text, original bytes, probability and requested alternatives (0–20); `refusal: null`. Hidden reasoning, successful tool-call syntax, EOS and wholly suppressed stop tokens are excluded. Malformed tool blocks that become visible text retain their logprobs. A token overlapping a stop boundary still describes the actual sampled token. |
| Reasoning | `none/minimal/low/medium/high/xhigh/max` resolve through the model frontend into native effort and/or thinking switches. Unsupported settings and conflicting explicit switches return errors. `/v1/models` exposes `reasoning.effort_mapping`. `reasoning_content` and template switches are DGPP extensions. |
| Request metadata | `store: false` (or omitted/null), text modality, bounded metadata, user and safety identifier accepted. Metadata is echoed on one-shot responses but not persisted. User/safety identifiers do not invoke an OpenAI safety or account service. `service_tier: auto/default` reports the actual `default` tier. |
| Image inputs | GLM-5.3-Flash user messages accept `image_url` parts containing PNG/JPEG data URIs. `/v1/models` reports `input_modalities`. See [image inputs](vision.md) for budgets and caching behavior. |
| Unsupported features | Stored-completion CRUD and `store: true`; remote image URLs, video/audio inputs and audio outputs; web search; predicted output; moderation configuration; explicit cache breakpoints/TTL/retention/key controls; verbosity; deprecated function-call API. Known unsupported API features return an error naming the field. |

DGPP also accepts existing extensions: `top_k`, `min_p`,
`repetition_penalty`, `prefix_cache`, `chat_template_kwargs`, object-valued
historical tool arguments, flat function definitions, and template-specific
tool output lists. These extensions should not be assumed portable to OpenAI.
Historical function arguments must be valid JSON objects for DGPP's templates.

## File inputs

User messages accept `{ "type": "file", "file": { "filename": "report.pdf",
"file_data": "data:application/pdf;base64,..." } }` or
`{ "type": "file", "file": { "file_id": "file-..." } }`.
Only one of `file_data` and `file_id` may be supplied. Data must be valid
base64; the data-URI prefix is optional. URLs and local filesystem paths are
not input-file references.

PDFs are processed with `pdftotext`, preserving extracted text and layout.
**PDF file inputs supply extracted text only; page images and diagrams are not interpreted.**
A PDF with no extractable text is rejected with `pdf_text_unavailable`; use
OCR before submitting a scanned document. UTF-8 text/code files (including
`.txt`, `.md`, `.json`, `.csv`, `.py` and `.cpp`) are supported as a DGPP
extension. OpenAI Chat Completions documents PDF file parts only.

Upload files using multipart `POST /v1/files` with `purpose=user_data` (or
`assistants`) and a `file` part. `GET /v1/files`, `GET /v1/files/{id}`,
`GET /v1/files/{id}/content` and `DELETE /v1/files/{id}` provide the input-file
lifecycle. These routes do not implement fine-tuning, batch uploads or
expiry policies. The list currently returns the complete store.

File work runs on bounded preprocessing workers; health, metrics and existing
streams remain available during PDF extraction. Prefill starts after file
processing, rendering and tokenization. Extracted text counts toward the
ordinary context/KV limit. Byte limits are independent of token capacity.
Shutdown stops active PDF extraction and drains pending file responses.

Configure the deployment's engine object (values below are the defaults,
except the example persistent directory):

```json
{
  "engine": {
    "file_inputs": {
      "directory": "/home/stephen/.cache/dgpp/files",
      "pdf_command": "pdftotext",
      "max_file_bytes": 52428800,
      "max_request_bytes": 52428800,
      "max_text_bytes": 134217728,
      "max_storage_bytes": 1073741824,
      "pdf_timeout_ms": 120000,
      "workers": 4
    }
  }
}
```

An omitted/empty directory creates an ephemeral store: file IDs expire when
that server process exits. A configured directory preserves uploads across
restarts. Delete files to reclaim the configured storage quota. Install
`poppler-utils` on rank 0 for PDF extraction. Inline base64 also counts toward
`http.max_body_bytes` and adds approximately one-third encoding overhead.

## Custom tools

Tools accept OpenAI's `{ "type": "custom", "custom": { "name": "code_exec",
"description": "Run code", "format": { "type": "text" } } }` shape. Format
is optional. Named selection and `allowed_tools` work with custom tools, and
function/custom tools may be mixed. Responses and streaming deltas use
`type: "custom"` and `custom: { name, input }`; assistant history accepts the
same shape. The client remains responsible for executing the tool.

DGPP adapts these declarations to a function with one `input` argument for
its existing checkpoint templates, and converts the result back to a custom
call. This argument is generated as a JSON string literal so quotes,
newlines and delimiters inside the free-form input survive transport.

Grammar format accepts `{ "type": "grammar", "grammar": { "syntax":
"regex", "definition": "[A-Z]+=[0-9]+" } }` or `syntax: "lark"`. Constraints
apply while decoding, not merely when validating the finished answer.
Lark supports named rules, literals, regular-expression terminals,
alternatives, groups, optional groups, repetition, right recursion, direct
non-nullable left recursion, `%ignore` and common terminal imports. Indirect
or hidden left recursion, nullable left recursion, and rule/terminal priorities
are rejected. Tree annotations do not affect the accepted language.
Unsupported directives and malformed grammars are errors.
Grammar formats require an engine with constrained decoding.

## Reasoning settings

| Model/template | Mapping |
| --- | --- |
| GLM effort-aware templates | minimal/low → low; medium/high → high; xhigh/max → max. `none` requires a native thinking switch and is rejected when thinking is always on. |
| Qwen effort-aware templates | minimal/low → low; medium → medium; high/xhigh/max → xhigh; none → `enable_thinking: false`. |
| DeepSeek-V4.1 | none → chat mode; minimal/low/medium/high/xhigh/max → native budgets 25/50/62/75/100/100. |
| Templates with only a thinking switch | none disables thinking; positive efforts enable it without inventing additional budget levels. |

Explicit `thinking`/`enable_thinking` settings must agree with
`reasoning_effort`. Hiding reasoning on the wire does not disable model
reasoning. `reasoning.effort_mapping` in `/v1/models` reports the mapping for
the loaded frontend.

## Structured output details

The grammar supports the documented OpenAI types, properties, required keys,
closed objects, array items and item bounds, enums, `anyOf`, numeric bounds,
local JSON Pointer references and recursion, `pattern`, `format`, and
`multipleOf`. Formats are date-time, time, date, duration, email, hostname,
ipv4, ipv6 and uuid. Pattern matching uses decoded string content (including
JSON escapes); numeric divisibility uses decimal arithmetic without rounding
the generated value to a floating-point number.

External schema references and unproductive reference cycles are rejected.
Array bounds must be integers from zero through 2,147,483,647; larger bounds
are rejected rather than truncated. Local pointers traverse object keys and
array indices, with JSON Pointer escaping and index validation.
Container-valued enums generate a canonical JSON spelling of an allowed value.
Constraints beside `$ref` and composition keywords outside this implementation
remain unsupported. Strict requests receive an
error instead of silently losing constraints. The new constraints travel in
the existing grammar journal to every rank. Token masks share vocabulary
prefixes through a trie and are checked against the exhaustive mask oracle.

Unknown top-level client/provider extensions are accepted and ignored, as in
earlier DGPP versions. This includes `preserveThinking`, which OpenCode can
forward from its model options. It does not control DGPP's prompt rendering;
use `chat_template_kwargs.clear_thinking` for the existing template control.
Recognized request fields remain validated, and known unsupported API features
still return explicit errors. Unknown fields inside `chat_template_kwargs`
remain errors because that object explicitly requests prompt changes.

Streaming uses `chat.completion.chunk` events followed by `[DONE]`.
With `stream_options.include_usage: true`, ordinary chunks carry `usage: null`
and one final chunk carries aggregate usage with `choices: []`. Interrupted
streams may have no final usage chunk. Content logprobs may follow content in
an event with an empty delta, because parser and stop buffers can delay their
visibility. Clients should accumulate content and logprobs independently.
`include_obfuscation` defaults to true and adds random padding to delta events;
false disables it. Padding is transport-only and consumes no model tokens.

Errors use `{ "error": { "message", "type", "param", "code" } }`.
An overload detected before opening a stream returns HTTP 503. Errors after
stream headers have been sent use an SSE error followed by `[DONE]`.
HTTP body-limit errors return 413 with `code: "request_too_large"`; this is a
serialized byte limit, separate from model context/KV capacity.

The legacy `/v1/completions` route serves string prompts and a narrower feature
set; it is not the full legacy API. Its streamed object type is
`text_completion`, with legacy logprobs fields. Model list/retrieve and health
endpoints remain available. Stored completion routes are not provided.

## Metrics and prefill progress

**OpenAI Chat Completions defines completion usage, not a metrics endpoint or
live prefill-progress event.** DGPP's `GET /metrics` and `GET /v1/metrics` are
equivalent operational extensions returning `application/json`. They do not
provide Prometheus exposition. Progress never appears as a fabricated chat
delta or usage event.

After rebuilding and redeploying this version, monitor rank 0:

```bash
watch -n 2 'curl -fsS http://192.168.50.221:18080/metrics | jq "{prefill, scheduler: (.scheduler | {active, queued, prefilling, snapshot_age_ms})}"'
```

`prefill.requests` contains one entry per currently prefilling scheduler choice:

```json
{
  "id": "chatcmpl-example",
  "slot": 0,
  "prompt_tokens": 282582,
  "processed_tokens": 165888,
  "cached_tokens": 0,
  "computed_tokens": 165888,
  "remaining_tokens": 116694,
  "elapsed_ms": 360000
}
```

- `processed_tokens` includes attached cached tokens plus newly computed tokens.
- `computed_tokens = processed_tokens - cached_tokens`.
- `remaining_tokens = prompt_tokens - processed_tokens`.
- The same token fields directly under `prefill` sum its current requests.
  They are gauges, not cumulative throughput counters; entries disappear when
  prefill finishes, is cancelled, or the engine fails.
- Publication happens at completed chunk boundaries. The HTTP thread reads
  an independent locked snapshot without reading mutable model state. No
  additional GPU synchronization is introduced. A slow chunk can leave
  progress unchanged until it completes; `elapsed_ms` continues increasing.
- `scheduler.active`, `queued` and `prefilling` include live service records.
  Other scheduler and prefix-cache counters are published after each scheduler
  pass. `scheduler.snapshot_age_ms` tells how old that snapshot is; a long
  synchronous prefill can leave pool and cumulative counters stale even while
  `prefill.requests` advances. Tokenization before admission is not prefill.
- `service.requests_total` counts validated HTTP generation requests reaching
  admission, once even when `n > 1`, including overload rejections. Shed,
  cancellation, pool exhaustion and failure counters each count a request at
  most once. `rejects_bad` counts service-generated 4xx responses, excluding
  transport-level parsing/size errors. Pending admissions/cancellations,
  scheduler counters and TTFT observations count individual choices.

KV capacity remains a shared token pool; concurrency does not divide it into
equal per-request partitions. Live progress, pool capacity, the HTTP body cap,
and a client's context/compaction/timeout settings are separate limits.

## Validation

Host tests exercise real HTTP/SSE transport with a deterministic engine:
nullable options and numeric bounds, unsupported capabilities, developer/text
messages, streaming usage and obfuscation, visible-content logprobs, multiple
choices, request counters and metrics while a synchronous prefill is blocked.
Parser tests cover buffered and malformed DSML token attribution. Scheduler,
fabric-serving and HTTP tests cover lifecycle and failure paths. The release
build compiles the GLM and shared session-model progress hooks. These checks
do not substitute for a multi-rank GPU run after deployment.

See the [edge-case validation record](openai-api-validation.md) for the
document/custom-tool/reasoning/schema regression matrix, fixes and commands.
