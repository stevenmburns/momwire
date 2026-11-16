"""Every short refusal token a C++ route can hand `not_served()` has a sentence.

A deck the executable's C++ route declines prints "not served by this build:"
and the route's reason. Most reasons are sentences already; the rest are
short names (`low-stand-off`, `stepping-fr-count`, ...) that
`standalone/momwire_razor2p.cpp`'s `not_served()` turns into words through its
`words[]` table. A name missing from the table printed raw: bs2 users saw
`front:bspline-low-stand-off` and `bspline: buried-capped-plan` (census
11-15).

The tokens are found by grep, in the sources the executables compile
(`standalone/CMakeLists.txt`'s `${MW_SRC}/*.cpp`, every header beside them,
and `standalone/*.cpp`), at every construct a decline reason is spelled in:

- a literal handed to a decline or not-ported exception (`Declined{"x"}`,
  `Decline{"x"}`, `PrintoutDecline{"x"}`, `FrontNotPorted("x")`,
  `decline("x")`), or assigned to a run outcome's `message`;
- a hyphenated literal returned as a front's `why` (razor's `scope` /
  `build`), or bound to a named constant;
- a `somm_trans::Refused` token (thrown, or returned by `eval_refusal` for
  its callers to throw), which bs2's buried fill wraps as `buried-<token>`,
  or as `buried-refusal: <token>` for what `cross_extents` throws.

A literal with a space is a sentence and is not looked up. `not_served()`
strips the `front:` prefix the seams add, then the `bspline-` / `bspline: `
route label bs2's front and fill add, before the lookup; `_lookup_key` does
the same here. A new token without a sentence fails this test by name.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "momwire"
EXE_MAIN = ROOT / "standalone" / "momwire_razor2p.cpp"

# Literals that are not decline reasons although a construct below matches:
# the executable's own argument names and its route-log labels.
NOT_REASONS = frozenset(
    {"declined:", "forced", "gmo", "unreadable", "unwritable", "variant", "front:"}
)

_LITERAL = re.compile(
    r"(?:\b(?:Declined|Decline|PrintoutDecline|FrontNotPorted)\s*[{(]\s*"
    r"|\bdecline\(\s*|\.message\s*=\s*)\"([^\"\\]+)\""
)
_RETURNED = re.compile(r"\breturn\s+\"([a-z0-9]+(?:-[a-z0-9]+)+)\";")
_CONSTANT = re.compile(
    r"\bconst\s+char\s*\*\s*\w+\s*=\s*\"([a-z0-9]+(?:-[a-z0-9]+)+)\";"
)
_REFUSED = re.compile(r"\bRefused\(\"([a-z0-9-]+)\"\)")
_TABLE_ENTRY = re.compile(r"\{\"([^\"]+)\",\s*\"[^\"]+\"\}")


def _sources() -> list[Path]:
    cmake = (ROOT / "standalone" / "CMakeLists.txt").read_text()
    cpp = {SRC / name for name in re.findall(r"\$\{MW_SRC\}/(\w+\.cpp)", cmake)}
    assert len(cpp) > 30, "the CMake source list moved; update this scan"
    return (
        sorted(cpp)
        + sorted(SRC.glob("*.h"))
        + sorted((ROOT / "standalone").glob("*.cpp"))
    )


def _body(text: str, signature: str) -> str:
    """The brace-matched body of the function whose definition starts with
    `signature`."""
    start = text.index("{", text.index(signature))
    depth = 0
    for k in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[start : k + 1]
    raise AssertionError(f"unbalanced braces after {signature}")


def _somm_refusals(text: str) -> tuple[set[str], set[str]]:
    """`somm_trans::Refused` tokens: every one (thrown, or returned by
    `eval_refusal`, which its callers throw), and those `cross_extents` throws,
    which the fill alone wraps as `buried-refusal: `."""
    thrown = set(_REFUSED.findall(text))
    every = thrown | set(
        _RETURNED.findall(_body(text, "std::string eval_refusal(const Grid &g"))
    )
    crossing = set(_REFUSED.findall(_body(text, "CrossPlan cross_extents(")))
    assert thrown and crossing, "no somm_trans::Refused tokens found; the scan is stale"
    return every, crossing


def _emitted() -> dict[str, str]:
    """token -> the file that spells it."""
    found: dict[str, str] = {}
    for path in _sources():
        text = path.read_text(errors="replace")
        if path.name == "_somm_transmitted_core.cpp":
            # reaches a route only inside bs2's buried fill, wrapped
            every, crossing = _somm_refusals(text)
            for token in every:
                found.setdefault(f"buried-{token}", path.name)
            # `buried-cross-grazing` is backed off from, not wrapped
            for token in crossing - {"buried-cross-grazing"}:
                found.setdefault(f"buried-refusal: {token}", path.name)
            continue
        for pattern in (_LITERAL, _RETURNED, _CONSTANT):
            for m in pattern.finditer(text):
                token = m.group(1)
                if " " not in token and token not in NOT_REASONS:
                    found.setdefault(token, path.name)
    return found


def _lookup_key(reason: str) -> str:
    """`not_served()`'s key for a reason: no `front:`, no bs2 route label."""
    if reason.startswith("front:"):
        reason = reason[len("front:") :]
    for label in ("bspline-", "bspline: "):
        if reason.startswith(label):
            return reason[len(label) :]
    return reason


def _table() -> set[str]:
    text = EXE_MAIN.read_text()
    start = text.index("std::string not_served(std::string why)")
    body = text[start : text.index("};", start)]
    return set(_TABLE_ENTRY.findall(body))


def test_the_scan_sees_the_known_tokens():
    emitted = _emitted()
    for token in (
        "low-stand-off",
        "buried-capped-plan",
        "stepping-fr-count",
        "phantom-threshold",
        "malformed-args",
        "buried-fill",
        "buried-transmitted-r-min-floor",
    ):
        assert token in emitted, token


def test_every_refusal_token_has_a_sentence():
    table = _table()
    missing = {
        token: where
        for token, where in sorted(_emitted().items())
        if _lookup_key(token) not in table
    }
    assert not missing, (
        "refusal tokens with no sentence in not_served()'s words table "
        f"(standalone/momwire_razor2p.cpp): {missing}"
    )


def test_the_census_spellings_find_their_sentences():
    table = _table()
    for reason in ("front:bspline-low-stand-off", "bspline: buried-capped-plan"):
        assert _lookup_key(reason) in table, reason
