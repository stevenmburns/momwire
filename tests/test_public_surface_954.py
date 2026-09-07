"""The README's `Public names` section is `__all__`, both directions (#954).

`__all__` is the source of truth; the README section is for a reader who wants
to know what may be imported without reading source. Two lists saying the same
thing is a second source of truth, and this repo has watched that shape fail
twice already -- momwire#848's two copies of the exemption test drifted apart,
and `_capabilities` refuses to let a consumer re-derive the axes for exactly
the same reason.

So the doc is not maintained by discipline, it is maintained by this test. A
name promoted without a line, or a line for a name that was never exported, is
a red test rather than a stale document.

Shaped after `test_makefile_lanes.py`: parse the artifact, compare with the
code, and report BOTH differences by name so the fix is obvious from the
failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import momwire

README = Path(__file__).resolve().parent.parent / "README.md"

# Names appear as the first backticked token(s) of a bullet, comma-separated:
#     - `A`, `B` — what they are.
_BULLET = re.compile(r"^- ((?:`[A-Za-z_][A-Za-z0-9_]*`(?:, )?)+)", re.M)
_NAME = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


def _section() -> str:
    text = README.read_text()
    start = text.index("## Public names")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def _documented() -> set[str]:
    return {n for m in _BULLET.finditer(_section()) for n in _NAME.findall(m.group(1))}


def test_the_section_exists_and_is_not_empty():
    """The precondition. A parser that silently matched nothing would make both
    directions below compare an empty set against itself and pass."""
    names = _documented()
    assert len(names) >= 20, sorted(names)


def test_every_exported_name_is_documented():
    missing = set(momwire.__all__) - _documented()
    assert not missing, (
        "exported but not in README's `Public names`:\n  "
        + "\n  ".join(sorted(missing))
    )


def test_every_documented_name_is_exported():
    extra = _documented() - set(momwire.__all__)
    assert not extra, "in README's `Public names` but not exported:\n  " + "\n  ".join(
        sorted(extra)
    )


def test_every_documented_name_actually_resolves():
    """`__all__` membership is a claim; this is the claim tested. A name in
    both lists that does not resolve would pass the two set comparisons."""
    for name in sorted(_documented()):
        assert getattr(momwire, name, None) is not None, name
