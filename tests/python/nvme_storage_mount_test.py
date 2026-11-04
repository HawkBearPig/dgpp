#!/usr/bin/env python3
"""Real storage placement checks in a private mount namespace (Linux).

Run: sudo python3 tests/python/nvme_storage_mount_test.py --binary build-ci/cache_storage_test
No model or GPU is used. All mounts and files belong to a temporary fixture;
the namespace disappears even if the driver is interrupted.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if not args.inside:
        if os.geteuid() != 0:
            print("SKIP: private mount tests require root; run with sudo")
            return 2
        return subprocess.run(["unshare", "--mount", "--propagation", "private", sys.executable,
                               str(Path(__file__).resolve()), "--binary", str(binary), "--inside"]).returncode
    mounts = []

    def mount(*flags):
        subprocess.run(["mount", *flags], check=True)
        mounts.append(flags[-1])

    with tempfile.TemporaryDirectory(prefix="dgpp-storage-mounts-") as directory:
        root = Path(directory)
        try:
            for name in ("source", "source/nested", "bind", "outside", "readonly", "tmpfs", "ramfs",
                         "lower", "upper", "work", "overlay", "space tab\tback\\line\n"):
                (root / name).mkdir(parents=True, exist_ok=True)
            mount("--bind", str(root / "source"), str(root / "bind"))
            mount("--bind", str(root / "outside"), str(root / "bind/nested"))
            mount("--bind", str(root / "source"), str(root / "readonly"))
            subprocess.run(["mount", "-o", "remount,bind,ro", str(root / "readonly")], check=True)
            mount("--bind", str(root / "source"), str(root / "space tab\tback\\line\n"))
            mount("-t", "tmpfs", "-o", "size=16m", "tmpfs", str(root / "tmpfs"))
            mount("-t", "ramfs", "ramfs", str(root / "ramfs"))
            mount("-t", "overlay", "overlay", "-o",
                  f"lowerdir={root / 'lower'},upperdir={root / 'upper'},workdir={root / 'work'}",
                  str(root / "overlay"))
            subprocess.run([str(binary)], check=True,
                           env={**os.environ, "DGPP_TEST_FILTER": "cache_storage_mount_namespace",
                                "DGPP_TEST_MOUNTS_ROOT": str(root)})
            # Preserve the source; remove only the private bind mounts.
            subprocess.run(["umount", str(root / "bind/nested")], check=True)
            mounts.remove(str(root / "bind/nested"))
            subprocess.run(["umount", str(root / "bind")], check=True)
            mounts.remove(str(root / "bind"))
            subprocess.run([str(binary)], check=True,
                           env={**os.environ, "DGPP_TEST_FILTER": "cache_storage_unmounted_namespace",
                                "DGPP_TEST_UNMOUNTED_ROOT": str(root)})
        finally:
            for target in reversed(mounts):
                subprocess.run(["umount", target], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
