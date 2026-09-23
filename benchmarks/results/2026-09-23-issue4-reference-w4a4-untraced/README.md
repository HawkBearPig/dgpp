# Untraced W4A4 reference — invalid control, aborted before verdict

The explicit `--moe-backend flashinfer_cutlass` option changed vLLM's cache
identity even though the backend matched the previous auto selection. Startup
used a new cache and retuned 17 of 42 tactics instead of using the retained
cache checked during preflight. See `invalid-control.json`.

The first request was interrupted before a response. No accuracy conclusion
is drawn. The full source/configuration, expected and effective caches, logs
and restoration receipt are retained in `raw/`. Both reference containers were
removed, and the exact production binary/configuration and inference were
independently verified after restoration.

A valid retry must retain the original default backend selection and verify
the runtime cache identity and tactics before issuing the request. Merely
checking that the expected cache file exists is insufficient.
