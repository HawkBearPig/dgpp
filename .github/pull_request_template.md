## Summary

What changes and why. Link the issue it closes.

## Validation

- Suites run, with the build they ran on. A `ctest` verdict counts only
  after a full rebuild of that tree.
- For anything under `src/net/`, the graph engine or the service's
  multi-rank paths: the fabric evidence, with every rank at zero swap,
  matching operation streams and the source revision and binary hash.
- For a performance change: decode tok/s by concurrency and cold prefill
  measured separately, before and after, and the affected rows of
  `docs/benchmarks.md` re-measured on the shipped template.
- For a numerics or representation change: the bitwise gate or the
  numerical accuracy check that judged it.

## Documentation

- [ ] The configuration reference and the deployment catalogue describe any
      new or changed `engine.*` key, with its default.
- [ ] The affected guide under `docs/` describes the current behaviour, in
      the present tense, without history.
- [ ] No benchmark results, profiles or run logs are added to the tree.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the build presets, the test
discipline and the code style.
