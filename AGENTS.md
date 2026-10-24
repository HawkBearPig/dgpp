# Working in this repository

Use the primary repository checkout for development. Do not create Git
worktrees or use linked checkouts to isolate tasks. Preserve existing work
when updating branches, and leave completed changes committed and pushed
when the user requests that outcome.

Keep generated benchmark results, profiles and run logs untracked. Do not
force-add ignored benchmark artifacts. Summarize validation in the PR and
keep reusable documentation in `docs/`, without links to local run artifacts.

Benchmarks follow the method of record in `docs/benchmarks.md` ("Workload and
timing"): one deployment template, measured as shipped, yields every row the
doc keeps for it, and `scripts/serve_campaign.py run --config <template>`
runs that set in one command. Report decode and prefill separately — decode
tokens/s by concurrency (the engine's decode steps only) as the throughput
figure, cold prefill in seconds, and the wall rate (prefill and queueing
included) only as the labelled service-level table. A row set is valid only
with its source revision and binary hash recorded, every rank at zero swap
on the binary under test, zero cached tokens in cold probes, matching
operation streams, an otherwise idle fabric (no build, bench or other world
during the measurements) and production restored once at the end. Fix a
defect a measurement exposes before publishing the row that exposed it.
