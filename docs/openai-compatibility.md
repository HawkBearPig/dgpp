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
| Image inputs | GLM-5.3-Flash user messages accept `image_url` parts containing PNG/JPEG/WebP data URIs. `/v1/models` reports `input_modalities`. See [image inputs](vision.md) for budgets and caching behavior. |
| Unsupported features | Stored-completion CRUD and `store: true`; remote image URLs, video/audio inputs and audio outputs; web search; predicted output; moderation configuration; explicit cache breakpoints/TTL/retention/key controls; verbosity; deprecated function-call API. Known unsupported API features return an error naming the field. |

DGPP also accepts existing extensions: `top_k`, `min_p`,
`repetition_penalty`, `prefix_cache`, `chat_template_kwargs`, object-valued
historical tool arguments, flat function definitions, and template-specific
tool output lists. These extensions should not be assumed portable to OpenAI.
Historical function arguments must be valid JSON objects for DGPP's templates.

Both completion endpoints accept the `ignore_eos` extension. It defaults to
`false` and accepts only a boolean; explicit `null` and other non-boolean
values return HTTP 400 naming `ignore_eos`. When `true`, sampled EOS tokens
still count toward usage but do not end generation. The token limit, stop
strings, cancellation and resource limits still apply.

## Streaming keep-alives (DGPP extension)

Both `POST /v1/chat/completions` and `POST /v1/completions` send SSE comments
while an accepted stream waits in the queue, prefills or pauses between output
chunks. The comment payload is `: keep-alive\n\n`, carried in the normal
chunked HTTP response. Comments never enter the completion content, token
usage, finish reason or `[DONE]` sequence. Multiple chat choices share one
keep-alive timer, and all comments stop when the stream ends or disconnects.

The server default is 30 seconds of silence. `http.sse_ping_interval` in the
cluster JSON changes it; `--sse-ping-interval` overrides JSON. A request with
`"stream": true` may set top-level `"sse_ping_interval": 15` to override either,
or `-1` to disable pings. Values must be integers in 1–2147483647 or `-1`;
invalid values, including explicit null, and use without `stream: true`
return HTTP 400 naming `sse_ping_interval` before admission.

SSE parsers ignore comment lines. Network read timeouts can be kept alive by
the bytes even if the parser hides them; an application timeout waiting for a
completion chunk still needs its own policy. Pings do not extend server
engine or shutdown deadlines. Optional `return_progress` / `prompt_progress`
events are not implemented by this extension.

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

## Function tool rendering

DGPP accepts both `{ "type": "function", "function": { "name": "lookup",
"parameters": { ... } } }` and flat `{ "name": "lookup", "parameters":
{ ... } }` definitions. Prompt rendering keeps `name`, `description`,
`parameters` and `strict`, the optional tool `type`, and their original
member order. Nested parameter schemas are preserved. Other tool metadata,
such as a function-level `response` schema, is omitted from the prompt.

DeepSeek also retains `namespace` on the tool or function so its rendered
names and descriptions agree with constrained decoding. Set
`DGPP_TOOLS_RAW=1` in the server environment before startup if a client
needs additional tool metadata rendered verbatim.

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
forward from its model options, and top-level `preserve_thinking` and
`preserve_reasoning`. They do not control prompt rendering, even when
`chat_template_kwargs` is absent.

History controls in `chat_template_kwargs` follow vLLM's native-key model:
`preserve_thinking`, `clear_thinking`, `drop_thinking`,
`truncate_history_thinking`, and `preserve_reasoning` pass through under their
original names. The checkpoint's template determines their meaning. DGPP does
not translate names, invert values, or reject different keys as contradictory.
In particular, `preserve_reasoning` is not a generic alias. A key the template
does not read has no effect. For Qwen3.8-Flash-Next, use
`{"chat_template_kwargs":{"preserve_thinking":false}}` to drop earlier history
reasoning according to its template's policy.

`--default-chat-template-kwargs JSON` sets defaults for the supported template
kwargs. Request kwargs override matching keys; omitted keys retain server
defaults, and keys absent from both use the checkpoint's behavior. For example,
server defaults `{"preserve_thinking":true,"clear_thinking":true}` plus request
kwargs `{"preserve_thinking":false}` render with `preserve_thinking=false` and
`clear_thinking=true`, independently of field order. `/v1/models` reports the
configured `default_chat_template_kwargs` object. Invalid defaults fail at
startup. Top-level request `reasoning_effort` also overrides its server default.

DGPP retains validation for its supported kwargs: history controls must be
booleans, duplicate keys are rejected, and conflicting explicit request
`reasoning_effort` / thinking controls still return errors. This change adopts
vLLM's history-key and server-default precedence, not its entire validation
surface; `null` and `"auto"` are not boolean history overrides.
Recognized request fields remain validated, and known unsupported API features
still return explicit errors. Unknown fields inside `chat_template_kwargs`
remain errors because that object explicitly requests prompt changes.

Streaming uses `chat.completion.chunk` events followed by `[DONE]`.
Each choice's assistant-role preamble arrives immediately before its first
output, or before its terminal chunk if the completion is empty. The legacy
route likewise delays its initial empty text chunk. HTTP headers and SSE
keep-alive comments may arrive while the request is queued or prefilling;
they are not completion events.
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
equivalent operational extensions returning `application/json`;
`GET /metrics/prometheus` renders the same counters in the Prometheus text
format with the latency distributions JSON cannot carry (see
[Prometheus exposition](#prometheus-exposition)). Progress never appears as a
fabricated chat delta or usage event.

After rebuilding and redeploying this version, monitor rank 0:

```bash
watch -n 2 'curl -fsS http://192.168.50.221:18080/metrics | jq "{prefill, scheduler: (.scheduler | {active, queued, prefilling, snapshot_age_ms})}"'
```

### Speculative decoding counters

`scheduler.spec_decode` exposes the engine's cumulative MTP verification
counters on both metrics routes:

| Field | Meaning |
| --- | --- |
| `depth` | Configured maximum speculative depth (0 for engines without MTP). |
| `num_drafts_total` | Request verification rounds that attempted at least one draft position. |
| `num_draft_tokens_total` | Sum of attempts across all speculative positions. |
| `num_accepted_tokens_total` | Sum of accepted draft tokens across all positions. |
| `num_draft_tokens_per_pos_total` | Attempt counts, starting with the first speculative position. |
| `num_accepted_tokens_per_pos_total` | Accepted counts in the same position order. |

The arrays contain `depth` entries, up to eight. Non-MTP engines report zero
totals and empty arrays. Counts exclude graph padding and the non-speculative
row. An attempt at a later position still counts when an earlier rejection
prevents accepting it. Counts include drafts accepted by the exact host fallback during sampled
decoding. These are final verification decisions before response stop and
length trimming, so accepted drafts need not all appear in the response.

Counters accumulate for the engine lifetime and reset when it is recreated.
They come from the existing completed-pass scheduler snapshot; use
`scheduler.snapshot_age_ms` to assess freshness. Read rank 0 once rather than
summing counters across tensor-parallel ranks.

For a measurement interval, subtract consecutive snapshots from the same
engine lifetime. Divide accepted tokens by attempted draft tokens for the
acceptance fraction, or by draft rounds for accepted drafts per round. Treat
a zero denominator as unavailable. For example, position attempts `[10, 8, 6]`
and accepts `[7, 4, 2]` mean 10 rounds, 24 attempted draft tokens and 13 accepted
drafts. Variable verification depth means attempted tokens are not necessarily
rounds multiplied by maximum depth.

For graph capacity and padded verification rows, see the
[decode graph batch counters](operations.md#decode-graph-batch-counters).
They describe launched graph work, while these counters describe completed
verification decisions; rejected drafts are not graph padding.

### Prometheus exposition

`GET /metrics/prometheus` returns `text/plain; version=0.0.4` for a
Prometheus scrape of rank 0 (set `metrics_path: /metrics/prometheus`). Every
sample carries `model_name`, the served model id. Families are `dgpp_`-prefixed;
counters end in `_total` and reset with the engine, which `rate()` handles.
Read rank 0 once: its counters already describe the whole world's work, and
the peers serve no HTTP.

The route's first three samples, `spec_decode_num_draft_tokens_total`,
`spec_decode_num_accepted_tokens_total` and `spec_decode_num_drafts_total`,
keep their original unprefixed, unlabeled form for existing scrapers;
`dgpp_spec_decode_*` carries the same counts with labels and per position.

The families mirror the JSON fields:

| Group | Families |
| --- | --- |
| Build and limits | `dgpp_build_info{version,git_sha,world_size,admission}` (always 1), `dgpp_engine_failed`, `dgpp_max_concurrent_requests`, `dgpp_queue_limit`, `dgpp_kv_pool_tokens`, `dgpp_admission_*_tokens` |
| Occupancy | `dgpp_num_requests_running` (slots held, prefills included), `_waiting`, `_prefilling`, `dgpp_pending_admissions`, `dgpp_pending_cancellations`, `dgpp_kv_pool_blocks_total`, `_in_use`, `dgpp_kv_cache_usage_perc` (0-1), `dgpp_scheduler_snapshot_age_seconds`, `dgpp_prefill_inflight_prompt_tokens`, `_remaining_tokens` |
| Requests | `dgpp_requests_total`, `_shed_total`, `_shed_pool_total`, `_cancelled_total`, `_failed_total`, `_rejected_total`, `dgpp_request_finished_total{reason}` (`eos`, `length`, `stop`, `cancelled`, `pool_exhausted`, `none`; counts choices), `dgpp_tool_calls_total`, `dgpp_reservations_grown_total` |
| Tokens and engine time | `dgpp_prompt_tokens_total` (attached prefixes included), `_computed_total`, `_cached_total`, `dgpp_generation_tokens_total`, `dgpp_prompts_prefilled_total`, `dgpp_prefill_seconds_total`, `dgpp_prefill_request_seconds_total`, `dgpp_decode_steps_total`, `dgpp_decode_rows_total`, `dgpp_decode_step_seconds_total` |
| Decode graphs | `dgpp_decode_batch_*` ([decode graph batch counters](operations.md#decode-graph-batch-counters)), `replays_by_slots_total{slots}` |
| Speculative decoding | `dgpp_spec_decode_depth`, `_num_drafts_total`, `_num_draft_tokens_total`, `_num_accepted_tokens_total`, `_num_draft_tokens_per_pos_total{position}`, `_num_accepted_tokens_per_pos_total{position}` (positions from 0) |
| Prefix cache | `dgpp_prefix_cache_slots`, `_entries`, `_blocks_pinned`, `_snapshot_bytes`, `_hits_total`, `_misses_total`, `_queries_total`, `_tokens_saved_total`, the snapshot, eviction, duplicate and skip counters (`_skipped_no_block_total` rising means pool pressure), the arena copy counts and seconds |

The histograms (`_bucket`, `_sum`, `_count`) are measured on rank 0's clock:

| Histogram | Interval |
| --- | --- |
| `dgpp_time_to_first_token_seconds{prefix_cache="hit"\|"miss"}` | Door to first token, split by whether the prompt attached to a cache entry. |
| `dgpp_request_queue_time_seconds` | Door to engine slot. |
| `dgpp_request_prefill_time_seconds` | Engine slot to first token. |
| `dgpp_request_decode_time_seconds` | First token to retire. |
| `dgpp_e2e_request_latency_seconds` | Door to retire. |
| `dgpp_request_time_per_output_token_seconds` | Decode time divided by the tokens after the first. |
| `dgpp_inter_token_latency_seconds` | The gap between engine passes that delivered a request's tokens; MTP lands several tokens per pass, so this is the pace a streaming client sees. |
| `dgpp_decode_step_duration_seconds` | Engine decode step time: each pass's mean, observed once per step. |
| `dgpp_request_prompt_tokens`, `dgpp_request_generation_tokens` | Request sizes in tokens. |

#### Per-rank families

The families above describe the service once. Each rank also reports its own
share of the work as `dgpp_rank_*`, labeled `rank`: `dgpp_rank_info{rank,world_size,version,git_sha}`,
`dgpp_rank_decode_steps_total`, `dgpp_rank_decode_step_seconds_total` and
`dgpp_rank_prefill_seconds_total` (wall time inside the engine's calls, the
collectives it waited on included), `dgpp_rank_generation_tokens_total`,
`dgpp_rank_num_requests_running`, `dgpp_rank_kv_pool_blocks_total` and
`_in_use`, `dgpp_rank_collectives_total` (the bus's completed multi-rank
collectives) and `dgpp_rank_snapshot_age_seconds`. Ranks run every step
together, so step time per step that is higher on one rank than on the others
points at that rank's GPU or link. Rank 0 appends its own (`rank="0"`) to
`/metrics/prometheus` in a multi-rank world. A peer serves no HTTP API; with
`DGPP_METRICS_PORT` set in `.env` (config `ports.metrics`, `--metrics-port`)
each peer serves its families at `GET /metrics/prometheus` on its node
address (an IPv4 address; `--metrics-bind` overrides it), adding
`dgpp_rank_ticks_total`, the journal records it applied. A peer publishes
after every tick it applies, so `dgpp_rank_snapshot_age_seconds` also grows
while the world is idle. The bus's per-lane traffic counters are not exported:
the bus updates them outside its stats lock.

Request histograms count choices, as the TTFT counters do, and observe at the
first token or at retire. A request shed before admission observes none of
them. Latency buckets run from 1 ms to 640 s, step buckets from 1 ms to 10 s and
token buckets from 1 to 1M.

### Prefill progress

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
