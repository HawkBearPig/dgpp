#!/bin/bash
# Build the release artifact: configure the
# release preset, installs the layout into dist/stage/dgpp-<version>/,
# writes the manifest with per-file checksums, and packs
# dist/dgpp-<version>.tar.zst. The version is the tree's
# (cmake/version.cmake): <base>+g<sha12>, ".dirty" when uncommitted.
#
#   scripts/release.sh            # build, stage, verify, pack; prints the tarball
#   scripts/release.sh --no-build # re-stage and pack the existing release build
#
# Layout inside the tarball (dgpp-<version>/):
#   bin/dgpp-serve                 the server, rpath $ORIGIN/../lib
#   lib/libcudart.so.13, lib/libcublas.so.13, lib/libcublasLt.so.13
#                                 the CUDA runtime it was built against
#   scripts/dgpp-cluster, scripts/site_env.py, scripts/serve_api_check.py
#   .env.example                   shared site settings (never credentials)
#   deploy/*.example.json          all model deployment templates
#   README.md, docs/               packaged setup and network/dependency guides
#   MANIFEST                       version, git sha, CUDA, host, date
#   MANIFEST.sha256                sha256 of every other file (sha256sum -c)
# A node also needs the driver, rdma-core, libnl, libstdc++ and the FFmpeg
# runtime libraries when video support is enabled.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/cluster_env.sh" || exit 1
BUILD=${DGPP_BUILD_DIR:-$ROOT/build-release}
DIST=$ROOT/dist
# Never turn a testing tree into a release tree, or label it as Release when
# --no-build is used. Select a separate directory with DGPP_BUILD_DIR.
if [ -f "$BUILD/CMakeCache.txt" ]; then
  BUILD_TYPE=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$BUILD/CMakeCache.txt")
  if [ "$BUILD_TYPE" != "Release" ]; then
    echo "release requires a Release build; $BUILD is ${BUILD_TYPE:-unconfigured}. Select build-release or another release directory with DGPP_BUILD_DIR." >&2
    exit 1
  fi
elif [ "${1:-}" = "--no-build" ]; then
  echo "no configured release build at $BUILD; run scripts/release.sh first" >&2
  exit 1
fi
if [ "${1:-}" != "--no-build" ]; then
  cmake --preset release -S "$ROOT" -B "$BUILD" >/dev/null
  cmake --build "$BUILD" -j --target dgpp_serve_app
fi
if grep -Eq '^DGPP_SANITIZE:STRING=.+' "$BUILD/CMakeCache.txt"; then
  echo "release cannot contain sanitizer instrumentation: $BUILD" >&2
  exit 1
fi
# Also check the artifact: --no-build may point at an older or custom build.
SECTIONS=$(readelf --wide --sections "$BUILD/dgpp-serve")
if printf '%s\n' "$SECTIONS" | grep -Eq '[[:space:]]\.(z?debug_[^[:space:]]*|gnu_debuglink)[[:space:]]'; then
  echo "release binary contains debug information: $BUILD/dgpp-serve; rebuild with the release preset" >&2
  exit 1
fi
VERSION=$(sed -n 's/^#define DGPP_VERSION "\(.*\)"$/\1/p' "$BUILD/generated/dgpp_version.hpp")
[ -n "$VERSION" ] || { echo "no version stamp in $BUILD/generated/dgpp_version.hpp"; exit 1; }
NAME=dgpp-$VERSION
STAGE=$DIST/stage/$NAME
rm -rf "$STAGE"; mkdir -p "$STAGE"
cmake --install "$BUILD" --prefix "$STAGE" >/dev/null
# The manifest: the identity, then every file's checksum.
{
  echo "name: $NAME"
  echo "version: $VERSION"
  echo "git: $(git -C "$ROOT" rev-parse HEAD)"
  echo "cuda: $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname) ($(uname -m))"
  echo "build_type: Release"
} > "$STAGE/MANIFEST"
(cd "$STAGE" && find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | sort | xargs sha256sum > MANIFEST.sha256)
# Verify before packing: the checksums, the version the binary reports, and
# that the binary resolves the CUDA runtime from its own lib/.
(cd "$STAGE" && sha256sum -c --quiet MANIFEST.sha256)
REPORTED=$("$STAGE/bin/dgpp-serve" --version)
case "$REPORTED" in *"$VERSION"*) ;; *) echo "version mismatch: binary says '$REPORTED', stamp is $VERSION"; exit 1 ;; esac
for lib in libcudart.so libcublas.so libcublasLt.so; do
  RESOLVED=$(ldd "$STAGE/bin/dgpp-serve" | awk -v l="$lib" '$1 ~ l {print $3}' | head -1)
  if [ -z "$RESOLVED" ] || [ "$(readlink -f "$RESOLVED")" != "$(readlink -f "$STAGE"/lib/$lib.*)" ]; then
    echo "the staged binary does not resolve $lib from its lib/ (got '$RESOLVED'):"; ldd "$STAGE/bin/dgpp-serve" | grep -E "cudart|cublas"; exit 1
  fi
done
mkdir -p "$DIST"
TARBALL=$DIST/$NAME.tar.zst
rm -f "$TARBALL"
tar -C "$DIST/stage" -I 'zstd -19 -T0' -cf "$TARBALL" "$NAME"
echo "$REPORTED"
echo "$(du -h "$TARBALL" | cut -f1)  $TARBALL"
