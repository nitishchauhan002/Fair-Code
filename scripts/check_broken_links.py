#!/usr/bin/env python3
"""Fail if a markdown in-page anchor or relative link points nowhere.

Checks every tracked `.md` file for:
  - `[text](#anchor)` links that don't match any heading on that same page
  - `[text](relative/path)` links to a file that doesn't exist
  - `[text](relative/path.md#anchor)` links where the anchor doesn't match
    any heading in that target file

External links (`http(s)://`, `mailto:`, ...) are out of scope - this is a
repo-integrity check, not a link-rot crawler, and making network calls in CI
would be slow and flaky.

The anchor slugifier below was reverse-engineered against every real anchor
link already in this repo (70/71 matched cleanly; the one holdout is a
literal `#link-to-section` placeholder inside a fenced code example in
CONTRIBUTING.md, which this script correctly ignores since it's in a fence)
rather than assumed from memory - GitHub's exact algorithm isn't public. The
rule that fits the evidence: lowercase, drop a hyphen that has a space on
both sides (a written dash, not a word-hyphen - this is what makes
"COMPAS - Criminal Justice Bias" slug as "compas--criminal-justice-bias"
while "Auto-Detection" keeps its hyphen), strip remaining punctuation, then
turn each remaining space into one hyphen. Treat this as a best-effort
approximation, not a guaranteed oracle - an unusual heading could still slip
past it in either direction.

Run locally:  python3 scripts/check_broken_links.py
Exit code:    0 = clean, 1 = at least one broken link found.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent

# faircode/_explainers/*.md is a generated, package-internal mirror of
# explainers/*.md (issue #388) - its files are byte-identical copies, so
# every relative link inside them is only valid relative to the ORIGINAL
# explainers/ directory the copy was made from, not the mirror's own,
# more-nested location. Checking the source in explainers/ already covers
# these links; checking the copy too would just be false positives.
ALLOW_PREFIXES = ("faircode/_explainers/",)

FENCE_RE = re.compile(r'^\s*```')
HEADING_RE = re.compile(r'^(#{1,6})\s+(.*?)\s*#*$')
LINK_RE = re.compile(r'(?<!!)\[[^\]]*\]\(([^)]+)\)')
INLINE_CODE_RE = re.compile(r'`[^`]*`')
INLINE_FORMATTING_RE = [
    (re.compile(r'\*\*([^*]*)\*\*'), r'\1'),
    (re.compile(r'\*([^*]*)\*'), r'\1'),
    (re.compile(r'\[([^\]]*)\]\([^)]*\)'), r'\1'),
]
EXTERNAL_SCHEME_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*:')


def _tracked_markdown_files():
    out = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    ).stdout
    return [
        ROOT / line for line in out.splitlines()
        if line.strip() and not line.startswith(ALLOW_PREFIXES)
    ]


def _strip_inline_formatting(text):
    for pattern, repl in INLINE_FORMATTING_RE:
        text = pattern.sub(repl, text)
    return text


def _slugify(heading_text):
    text = _strip_inline_formatting(heading_text).lower()
    # a hyphen flanked by spaces is a written dash, not a word-hyphen - drop it
    text = re.sub(r'(?<= )-(?= )', '', text)
    text = re.sub(r'[^\w\- ]', '', text, flags=re.UNICODE)
    return text.replace(' ', '-')


def _parse_file(path):
    """Returns (heading_slugs, [(line_no, raw_link_target), ...]), both with
    fenced code blocks excluded."""
    text = path.read_text(encoding="utf-8")
    heading_slugs = set()
    slug_counts = {}
    links = []
    in_fence = False

    for line_no, line in enumerate(text.splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        m = HEADING_RE.match(line)
        if m:
            slug = _slugify(m.group(2))
            heading_slugs.add(slug)
            if slug in slug_counts:
                slug_counts[slug] += 1
                heading_slugs.add(f"{slug}-{slug_counts[slug]}")
            else:
                slug_counts[slug] = 0
            continue

        scan_line = INLINE_CODE_RE.sub('', line)
        for link_m in LINK_RE.finditer(scan_line):
            raw = link_m.group(1).strip()
            if not raw:
                continue
            # split off a trailing "title" (`(path "some title")`), if present
            space_quote = re.search(r'\s+["\']', raw)
            target = raw[:space_quote.start()] if space_quote else raw
            links.append((line_no, target))

    return heading_slugs, links


def main():
    files = _tracked_markdown_files()
    headings_by_file = {}
    links_by_file = {}

    for path in files:
        try:
            headings_by_file[path], links_by_file[path] = _parse_file(path)
        except (OSError, UnicodeDecodeError):
            continue

    broken = []
    for path, links in links_by_file.items():
        for line_no, target in links:
            if not target or target == "#":
                continue
            if EXTERNAL_SCHEME_RE.match(target):
                continue

            if target.startswith("#"):
                anchor = target[1:]
                if anchor not in headings_by_file.get(path, set()):
                    broken.append((path, line_no, target, "no matching heading on this page"))
                continue

            rel_path, _, anchor = target.partition("#")

            # GitHub renders templates like .github/PULL_REQUEST_TEMPLATE.md
            # outside normal repo browsing, so they use a "../blob/<branch>/"
            # permalink-style relative path that isn't a real filesystem path.
            if re.search(r'(^|/)blob/[^/]+/', rel_path):
                continue

            resolved = (path.parent / unquote(rel_path)).resolve()

            if not resolved.exists():
                broken.append((path, line_no, target, f"{rel_path!r} does not exist"))
                continue

            if anchor and resolved.suffix == ".md" and resolved in headings_by_file:
                if anchor not in headings_by_file[resolved]:
                    broken.append(
                        (path, line_no, target, f"no heading {anchor!r} in {rel_path}")
                    )

    if broken:
        print("Broken internal links found:")
        for path, line_no, target, reason in broken:
            print(f"  {path.relative_to(ROOT)}:{line_no}: [{target}] - {reason}")
        return 1

    print(f"OK: {sum(len(v) for v in links_by_file.values())} internal links checked "
          f"across {len(files)} markdown files, none broken.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
