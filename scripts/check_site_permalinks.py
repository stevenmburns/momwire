#!/usr/bin/env python3
"""Verify the primer site's pinned momwire source permalinks.

The primer's rule (docs/mom-primer-plan.md): every source link in the site
content is pinned to a release tag, and every LINE-anchored link must have
an entry in site/permalinks.json giving the substring expected on that
line. This script re-derives each anchored line range from `git show
<tag>:<path>` and fails if the expectation no longer matches — so a
refactor between releases can't silently strand the prose.

Unanchored blob links are checked for path existence only (at the tag for
v* refs, in the working tree for blob/main refs, since those land with the
same PR that adds them).

Run from the repo root: python scripts/check_site_permalinks.py
No dependencies beyond git + stdlib.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "site" / "src" / "content"
MANIFEST = ROOT / "site" / "permalinks.json"

LINK_RE = re.compile(
    r"https://github\.com/stevenmburns/momwire/blob/"
    r"(?P<ref>[\w.\-]+)/(?P<path>[^#()\s\"'\]]+)"
    r"(?:#L(?P<l0>\d+)(?:-L(?P<l1>\d+))?)?"
)


def git_show(ref: str, path: str) -> list[str] | None:
    """Lines of `path` at `ref`, fetching the tag shallowly if missing
    (CI checkouts are often depth-1 without tags)."""
    for attempt in (0, 1):
        proc = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.splitlines()
        if attempt == 0 and ref.startswith("v"):
            subprocess.run(
                ["git", "fetch", "--depth=1", "origin", "tag", ref],
                cwd=ROOT,
                capture_output=True,
            )
    return None


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    manifest.pop("_comment", None)
    seen: set[str] = set()
    failures: list[str] = []
    n_links = 0

    for page in sorted(CONTENT.rglob("*.md*")):
        text = page.read_text()
        for m in LINK_RE.finditer(text):
            n_links += 1
            ref, path = m.group("ref"), m.group("path")
            url = m.group(0)
            where = f"{page.relative_to(ROOT)}: {url}"

            if ref == "main":
                if m.group("l0"):
                    failures.append(
                        f"{where}\n  line-anchored links must pin a tag, not main"
                    )
                elif not (ROOT / path).exists():
                    failures.append(f"{where}\n  {path} not in the working tree")
                continue

            lines = git_show(ref, path)
            if lines is None:
                failures.append(f"{where}\n  {path} does not exist at {ref}")
                continue
            if not m.group("l0"):
                continue

            l0 = int(m.group("l0"))
            l1 = int(m.group("l1") or l0)
            expect = manifest.get(url)
            seen.add(url)
            if expect is None:
                failures.append(f"{where}\n  no entry in site/permalinks.json")
                continue
            window = "\n".join(lines[l0 - 1 : l1])
            if expect not in window:
                failures.append(
                    f"{where}\n  expected {expect!r}\n  at L{l0}-L{l1} but found {window!r}"
                )

    for url in sorted(set(manifest) - seen):
        failures.append(f"site/permalinks.json: stale entry (no page links it): {url}")

    if failures:
        print(f"permalink check FAILED ({len(failures)} problem(s)):\n")
        print("\n\n".join(failures))
        return 1
    print(f"permalink check OK: {n_links} momwire links, {len(seen)} line-anchored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
