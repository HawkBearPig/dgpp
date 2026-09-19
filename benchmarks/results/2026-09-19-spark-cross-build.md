# Spark cross-build validation

The cross-build adds an Ubuntu 24.04 Docker environment, a CMake toolchain and
release-derived preset, and a small Bash wrapper for building ARM64/GB10
binaries on x86 Linux. It changes no inference code or deployment settings.

Base: upstream `a0708b969650ae1a8a65b8a1877fdcd45ce76590`. The original tooling
was isolated from `d272609`; the release preset retains the base's server-only
target and optimization/stripping settings.

## Reproduction

Run from a fresh checkout on x86 Linux with Docker:

```bash
docker build --no-cache --platform linux/amd64 -t dgpp-spark-cross:upstream-pr -f dev/Dockerfile.spark-cross dev
DGPP_CROSS_IMAGE=dgpp-spark-cross:upstream-pr scripts/spark-cross build
DGPP_CROSS_IMAGE=dgpp-spark-cross:upstream-pr scripts/spark-cross command cmake --install build-spark-cross --prefix "$PWD/install/spark"
python3 tests/python/spark_cross_test.py
python3 -m unittest discover -s tests/python -p '*_test.py'
```

## Dependency handling

CUDA's cross cuBLAS package contains a link stub. Install staging needs the
actual ARM64 cuBLASLt shared library. The Dockerfile extracts that library
from the ARM64 runtime package because installing both packages introduces
overlapping cuBLAS files. PCRE2 uses upstream's pinned static source build.
Package repository versions are not locked; record the installed versions
when reproducing this build.

Docker 28.2.2 built the clean image with CMake 3.28.3, AArch64 GCC/G++ 13.3.0, nvcc 13.0.88,
Python 3.12.3, cudart 13.0.96, cuBLASLt 13.1.1.3 and ARM64 libibverbs
50.0-2ubuntu0.2. Its local image ID was
`sha256:2b4256500b6eca04e1e561b9f460196a59cde5aa39bb143e9fd7fa81e7d42954`.

Linked worktrees need their common Git directory mounted read-only as well
as the source directory; otherwise the normal version target stamps the
binary as unknown. A real container probe resolves the source revision with
that mount. The wrapper regression covers its paths and access mode.

## Validation

- Clean Docker image build without cached layers passed.
- Ten wrapper tests cover CLI validation, argument boundaries, build failure
  propagation, missing images, ownership and linked-worktree metadata mounts.
- The workstation Python suite and the Python suites registered with CTest
  have two failures already present on unchanged upstream `a0708b9`:
  `PortabilityTest.test_scripts_index_and_setup_document_links` (upstream's
  missing `vision_prefix_cache_check.py` index entry) and
  `SiteEnvTest.test_cpp_fixture_matches_resolved_example` (prefill settings
  differ from the checked-in resolved fixture). The new script is indexed.
  Baseline discovery ran 150 tests with two failures and three skips.
  Final candidate discovery ran 160 tests: 155 passed, the same two failed,
  and three were skipped.
- Container CTest ran nine Python suites: seven passed, including all ten
  cross-build tests; two failed for the same baseline issues. These tests
  require a passwd entry for the numeric host UID, which the build image
  intentionally does not create. The test invocation mounted a temporary
  passwd file containing a `builder` entry at `/etc/passwd:ro`; ordinary
  builds need no passwd mapping. Running without that test fixture also
  exposes unrelated `getpwuid()` errors in the existing launcher tests.

- The server built successfully from an empty `build-spark-cross` directory,
  including PCRE2 10.45's C sources with the ARM64 compiler. Rebuilding with
  linked-worktree metadata visible restamped and relinked the server.
- The documented `cmake --install` command passed. `file` and `readelf`
  identify the server, cudart and cuBLASLt as AArch64 ELF files. The server is
  stripped with no `.debug_info`, retains `$ORIGIN/../lib` as its install
  RUNPATH, and needs `libcudart.so.13` and `libcublasLt.so.13`. There is no
  dynamic PCRE2 dependency. PCRE2's license is staged, while standalone
  PCRE2 archives, headers and tools are excluded from the runtime layout.
- Shell syntax, preset JSON syntax and `git diff --check` passed.

Local logs and package checks are retained under the ignored
`artifacts/cross-build-pr/` directory. No ARM64 executable was run on x86.

## Scope

No SSH, deployment, service restart or Spark GPU/RDMA work is part of this
preparation. Cross-compilation and ELF inspection do not establish runtime
correctness. Fresh native host C++ tests, target execution, and the full
GPU/RDMA suite remain outstanding on idle Spark hardware. The contribution
guide's full-build/full-suite merge gate has not been satisfied by host-only
wrapper tests. No inference performance or numerical change is claimed.
