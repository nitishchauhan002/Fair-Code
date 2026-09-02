#!/usr/bin/env python3
"""Fails if a "current explainer count" mention in README.md, CONTRIBUTORS.md,
METRICS.md, or ROADMAP.md drifts from the real count of explainers/*.md.

CHECKS below is a small, explicit list of exact regexes, one per file - not a
fuzzy "N explainers" scan across the whole file. METRICS.md in particular is a
weekly log full of legitimate historical counts (e.g. "explainer count `39 ->
44`" from an old week); matching those against today's count would be a false
positive worse than the drift this script is meant to catch. Each regex below
targets the one line in its file that's meant to state the current, live
total.

Run locally:  python3 scripts/check_explainer_count.py
Exit code:    0 = all mentions match, 1 = at least one is stale or missing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPLAINERS_DIR = ROOT / "explainers"

# (relative path, regex whose sole capture group is the claimed count)
CHECKS = [
    ("README.md", re.compile(r"Show all (\d+) explainers")),
    ("CONTRIBUTORS.md", re.compile(r"the bulk of the (\d+) explainers")),
    ("METRICS.md", re.compile(r"Explainers-(\d+)-blueviolet")),
    ("ROADMAP.md", re.compile(r"(\d+) explainers published")),
]


def main() -> int:
    actual = len(list(EXPLAINERS_DIR.glob("*.md")))
    stale = []

    for rel_path, pattern in CHECKS:
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        if match is None:
            stale.append((rel_path, f"expected pattern not found: {pattern.pattern!r}"))
            continue
        claimed = int(match.group(1))
        if claimed != actual:
            stale.append((rel_path, f"says {claimed} explainers, but explainers/*.md has {actual}"))

    if stale:
        print(f"Explainer count drift found (explainers/*.md currently has {actual} files):",
              file=sys.stderr)
        for rel_path, reason in stale:
            print(f"  {rel_path}: {reason}", file=sys.stderr)
        return 1

    print(f"OK: all {len(CHECKS)} explainer-count mentions match explainers/*.md ({actual} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
