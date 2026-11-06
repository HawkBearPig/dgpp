# Heretic integration

Heretic owns scoring plugins, modifier algorithms, optimization studies, and
exports. DGPP supplies execution for its supported checkpoint formats and will
apply the resulting runtime artifacts alongside the original weights.

The first integration increment provides a GLM-5.3 projection catalog and a
read-only inspection command. Model execution, activation capture, candidate
installation, and serving-time adapter loading remain to be implemented.

## Configuration and performance isolation

Set `engine.heretic_research` to `true` in a separate research cluster config
to use the inspection command. The default is `false`; existing deployment
templates are unchanged. This flag currently authorizes only the separate
research tool. It has no effect on `dgpp-serve` execution or its configuration
digest because serving performs no research operations.

Keep future capture buffers, tensor copies, synchronizations, trial controls,
and research listeners behind this flag. Before adding distributed execution,
carry its effective state through the rank journal and configuration digest.
Runtime adapter execution in ordinary serving will require a separate explicit
configuration opt-in. Disabled features must not allocate their buffers,
enqueue their kernels or collectives, or alter the existing cache policy.

## Inspect the target projections

Build the tool with:

```sh
cmake --preset release
cmake --build build-release --target dgpp_heretic_inspect -j 4
```

Use a resolved cluster JSON with explicit `nodes`, as produced by the cluster
launcher. A local config may instead use `world_size: 1`. Set the research flag
in that config and run this on a node that has the checkpoint:

```sh
build-release/dgpp-heretic-inspect --config /path/to/cluster.resolved.json --rank 0
```

The command resolves the configured model using the local Hugging Face cache
and the selected rank's `HF_HUB_CACHE` override. An explicit model directory
must contain `config.json`; a cache repository root is not a snapshot directory.
Inspection reads configuration and quantization sidecars, without opening
weight shards, initializing CUDA, or connecting to peers. The rank argument
selects the partition description; it does not launch that rank.

The JSON response identifies schema `dgpp.heretic.targets`, version 1, and
capability `describe_targets`. It includes the resolved checkpoint, rank,
world size, layer count, and targets in Heretic capture order. Each target
has a stable `layer/component/module_index` ID, canonical checkpoint module,
full matrix dimensions, local input-column interval, and packed weight format.
This capability does not advertise inference or adapter support.

For the full 78-layer Mixed346 checkpoint, the catalog contains 78 attention
output projections, three dense MLP down projections, and 75 shared-expert down
projections. Fused routed experts and the draft block are excluded. The shared
expert uses Heretic's logical `mlp.down_proj` component, with its distinct
checkpoint module path retained in the descriptor.

## Validation

Host unit tests check the target set, weight formats, rank bounds, and complete
nonoverlapping input partitions for worlds of size 1, 2, and 4. The command test
uses a directory containing only a tiny model config, with CUDA hidden, to
check the flag, JSON escaping, rank errors, and four partition descriptions.

```sh
cmake --build build-release --target unit_tests dgpp_heretic_inspect -j 4
ctest --test-dir build-release -R '^(unit_tests|heretic_inspect_test)$' --output-on-failure
```
