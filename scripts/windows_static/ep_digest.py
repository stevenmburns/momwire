"""A digest of preprocessed C++ that ignores layout (windows-static.yml's `msvc` job).

    python scripts/windows_static/ep_digest.py FILE

Prints the SHA-256 of FILE's non-blank lines with each run of whitespace
collapsed. `cl /EP` keeps the source's line structure as blank lines, so a
comment or a directive continued over two lines moves the raw bytes without
changing a single token the compiler reads. Any change to the tokens, or to
where a line breaks between them, changes the digest.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def digest(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    kept = "\n".join(line for line in lines if line)
    return hashlib.sha256(kept.encode("utf-8", "surrogateescape")).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    raw = Path(argv[0]).read_bytes()
    print(digest(raw.decode("utf-8", "surrogateescape")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
