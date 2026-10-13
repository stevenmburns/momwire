"""The deck population of the momwire#1454 step-3 gates.

Three sources, each deduplicated on content (sha256), in a stable order:

* momwire's own ``tests/fixtures/**/*.nec``;
* antennaknobs' EZNEC and 4nec2 captures (``scratch/eznec-capture``,
  ``scratch/4nec2-capture``): every ``.nec`` there, ``pre``/``post`` copies
  of one deck counted once;
* Ward's bench decks (``~/wardbench/decks``, not in either repository).

A source that is absent contributes nothing and is reported as absent.
"""

import hashlib
import os
from pathlib import Path

MOMWIRE = Path(__file__).resolve().parents[2]
SOURCES = (
    ("fixtures", MOMWIRE / "tests" / "fixtures"),
    ("eznec-capture", MOMWIRE.parent / "scratch" / "eznec-capture"),
    ("4nec2-capture", MOMWIRE.parent / "scratch" / "4nec2-capture"),
    (
        "wardbench",
        Path(os.environ.get("WARDBENCH", "~/wardbench")).expanduser() / "decks",
    ),
)


def decks():
    """[(source, path, sha256)], unique on content across all sources."""
    seen, out, absent = set(), [], []
    for name, root in SOURCES:
        if not root.is_dir():
            absent.append(name)
            continue
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".nec")
        for p in paths:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            out.append((name, p, digest))
    return out, absent


if __name__ == "__main__":
    found, absent = decks()
    by = {}
    for name, _p, _d in found:
        by[name] = by.get(name, 0) + 1
    print(len(found), by, "absent:", absent)
