# JSON mask CPU replay

`DGPP_JSON_FREE_STRING_MASK` is a CMake option, default `OFF`. Leaving it
unspecified preserves the existing masking algorithm. Enabling it lets an
unconstrained JSON value string use the existing table-mask path even when an
unrelated field has a string or numeric constraint. Use the same setting on
every serving rank. There is no deployment JSON or request field to set.

This targets schemas mixing prose with constrained fields such as a date.
It does not speed up matching inside a constrained string. Keys, escape states,
container enums and any constrained live alternative still use the trie.
Tokens that close the free string and enter later fields are checked by the
existing suffix simulation. No regular-expression settings or matching rules
change. The enabled build also drops unused table-mask answers from the
temporary root of a trie traversal, retaining the original machine's cache.

## Build and check both settings

Use the normal CUDA 13 build prerequisites, even though this executable only
runs host code. Separate build directories keep the controls explicit:

```bash
cmake -S . -B build-mask-off -DCMAKE_BUILD_TYPE=Release \
  -DDGPP_JSON_FREE_STRING_MASK=OFF -DDGPP_BUILD_BENCHMARKS=ON
cmake -S . -B build-mask-on -DCMAKE_BUILD_TYPE=Release \
  -DDGPP_JSON_FREE_STRING_MASK=ON -DDGPP_BUILD_BENCHMARKS=ON
cmake --build build-mask-off -j4 --target unit_tests json_mask_bench
cmake --build build-mask-on -j4 --target unit_tests json_mask_bench
DGPP_TEST_FILTER=json_grammar CUDA_VISIBLE_DEVICES='' build-mask-off/unit_tests
DGPP_TEST_FILTER=json_grammar CUDA_VISIBLE_DEVICES='' build-mask-on/unit_tests
DGPP_TEST_FILTER=tool_grammar CUDA_VISIBLE_DEVICES='' build-mask-off/unit_tests
DGPP_TEST_FILTER=tool_grammar CUDA_VISIBLE_DEVICES='' build-mask-on/unit_tests
```

Run the normal host suite as well; the filters above are the focused checks.
The native tests compare exact mask words, vocabulary size and allowed count
against exhaustive token simulation. They exercise boundary-crossing tokens,
subsequent field constraints, repeated masks, copied/restored machines, JSON
escapes, mixed alternatives, container enums and wrapper EOS/marker handling.

## Measure

Pass a local `tokenizer.json`; no model weights are loaded. Keep the compiler,
optimization, PCRE2 library, CPU affinity and tokenizer identical between builds.
Run serially on an otherwise idle CPU, alternating order across rounds:

```bash
taskset -c 0 build-mask-off/json_mask_bench /path/to/tokenizer.json 1 3 > off.csv
taskset -c 0 build-mask-on/json_mask_bench /path/to/tokenizer.json 1 3 > on.csv
taskset -c 0 build-mask-on/json_mask_bench /path/to/tokenizer.json 4 3 > on-rows4.csv
taskset -c 0 build-mask-off/json_mask_bench /path/to/tokenizer.json 4 3 > off-rows4.csv
```

The constructed fixtures are ordinary prose with a constrained date, the same
document without a date constraint, a short mixed document, and an active
anchored pattern. Every position calls `GrammarState::mask`, checks the actual
replay token and calls `allows`/`advance`. With multiple rows it also copies the
state, advances draft tokens, masks later rows and copies the mask words into
host staging rows, following the graph engine's staging sequence. EOS is part
of each replay. The tokenizer's highest ID is reserved as the replay's EOS
sentinel; model generation configuration is not loaded. This is a CPU
reproduction of the staging sequence, not an engine or GPU execution test.

CSV output reports construction, mask/staging, allows/advance and total replay
microseconds. Total includes bookkeeping, allocations, mask checks and hashing;
it excludes tokenizer/vocabulary/table preparation, document encoding, output
printing and end-of-round state destruction. Temporary draft copies are
destroyed within the measured mask/staging interval. Preparation is reported
separately or performed before the replay. Each replay constructs a
fresh state; later positions exercise its warm cache. The first round is
retained. Mask digests should match across builds for the same tokenizer,
fixture and row count; digests are a diagnostic, not an exact-mask oracle.

Report complete-document totals alongside mask time, short/plain/active controls
and first-use table cost. Host savings do not establish serving throughput,
model speed, speculative acceptance, multi-rank parity or workload prevalence.
Retain raw measurements locally; summarize source, toolchain, tokenizers,
failures and skipped hardware checks in the PR.
