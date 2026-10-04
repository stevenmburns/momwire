"""The release jobs download only what ships into ``dist/``.

v0.71.0's PyPI upload failed whole: ``download-artifact`` with no ``pattern``
merged every artifact of the run into ``dist/``, including the per-test
durations #1308 uploads, and the publish action refuses a directory holding a
file it cannot classify (``InvalidDistribution: Unknown distribution format:
'durations.jsonl'``). The wheels themselves had all passed.

Greps rather than parsing YAML, as ``test_release_notes.py`` does and for the
same reason.
"""

from __future__ import annotations

import pathlib
import re

WHEELS = (
    pathlib.Path(__file__).resolve().parent.parent
    / ".github"
    / "workflows"
    / "wheels.yml"
)


def _dist_downloads(text: str) -> list[str]:
    """Each ``download-artifact`` step's ``with:`` block that writes ``dist``."""
    steps = re.split(r"\n\s*- (?:uses|name):", text)
    return [
        s
        for s in steps
        if "actions/download-artifact" in s and re.search(r"path:\s*dist\b", s)
    ]


def test_every_dist_download_names_what_ships():
    blocks = _dist_downloads(WHEELS.read_text())
    # The GitHub-release attach and the PyPI publish.
    assert len(blocks) == 2
    for b in blocks:
        m = re.search(r'pattern:\s*"?([^"\n]+)"?', b)
        assert m, "a dist/ download with no pattern merges every artifact"
        assert "durations" not in m.group(1)
        assert m.group(1).strip() == "{wheels-*,sdist}"
