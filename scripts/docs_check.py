#!/usr/bin/env python3
"""Check the repository's Markdown: relative links resolve and the README
keeps its style.

Every tracked Markdown file is scanned for relative links. A link must
name a file or directory that exists, and a fragment on a Markdown target
must name a heading in that file (GitHub's slug rules) or a line, L<n>. README.md must
also contain no em-dash, en-dash or semicolon.

Run from the repository root. Exit status 1 lists every failure.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
STYLE = {"—": "em-dash", "–": "en-dash", ";": "semicolon"}


def tracked_markdown(root: str) -> list[str]:
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], check=True, capture_output=True).stdout
    return sorted(p for p in out.decode().split("\0") if p.endswith(".md"))


def slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def headings(path: str) -> set[str]:
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    in_fence = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = HEADING.match(line)
            if not m:
                continue
            base = slug(m.group(1))
            n = seen.get(base, 0)
            seen[base] = n + 1
            anchors.add(base if n == 0 else f"{base}-{n}")
    return anchors


def main() -> int:
    root = os.getcwd()
    failures: list[str] = []
    heading_cache: dict[str, set[str]] = {}
    for rel in tracked_markdown(root):
        path = os.path.join(root, rel)
        text = open(path, encoding="utf-8").read()
        for m in LINK.finditer(text):
            target = m.group(1)
            if re.match(r"^(https?:|mailto:|ftp:)", target):
                continue
            file_part, _, fragment = target.partition("#")
            if file_part:
                full = os.path.normpath(os.path.join(os.path.dirname(path), file_part))
                if not os.path.exists(full):
                    failures.append(f"{rel}: missing target {target}")
                    continue
            else:
                full = path
            if fragment and re.match(r"^L\d+(-L\d+)?$", fragment):
                continue  # a line reference, which GitHub resolves on any file
            if fragment and full.endswith(".md") and os.path.isfile(full):
                if full not in heading_cache:
                    heading_cache[full] = headings(full)
                if fragment.lower() not in heading_cache[full]:
                    failures.append(f"{rel}: no heading for {target}")
    readme = os.path.join(root, "README.md")
    with open(readme, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            for ch, name in STYLE.items():
                if ch in line:
                    failures.append(f"README.md:{lineno}: {name}")
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)} failure(s)")
        return 1
    print("docs_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
