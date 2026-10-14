"""Reading two printouts of one deck cell by cell (python-free arc, step 0).

`moved_cells` is the review artifact: every printed number that differs
between a golden printout and a fresh render, with how far it moved in units
of its own last printed digit. `scripts/golden_printouts.py --check` prints it
and `tests/test_golden_printouts.py` fails with it.

A printout is compared as lines of whitespace-separated tokens. Two lines
whose tokens differ in anything but a number, or two printouts whose line
counts differ, are a STRUCTURAL move: no tolerance applies to layout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([Ee][-+]?\d+)?$")
# NEC prints a three-digit exponent Fortran-style, dropping the E: 1.75281-107.
_THREE_DIGIT = re.compile(r"^([-+]?\d+\.\d+)([-+]\d{3})$")


def _canonical(token: str) -> str:
    m = _THREE_DIGIT.match(token)
    return f"{m.group(1)}E{m.group(2)}" if m else token


def number(token: str) -> Decimal | None:
    """The printed value of a numeric token, or ``None`` for text."""
    token = _canonical(token)
    if not _NUMBER.match(token):
        return None
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def last_digit(token: str) -> Decimal:
    """The value of one unit in the token's last printed digit."""
    mantissa, _, exponent = _canonical(token).upper().partition("E")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return Decimal(1).scaleb(-decimals + (int(exponent) if exponent else 0))


@dataclass(frozen=True)
class Move:
    line: int  # 1-based
    column: int  # 0-based token index
    golden: str
    rendered: str
    digits: (
        Decimal | None
    )  # |difference| in units of the last printed digit; None = text

    def __str__(self) -> str:
        how = "TEXT" if self.digits is None else f"{self.digits:.3g} last-digit units"
        return f"line {self.line} cell {self.column}: {self.golden} -> {self.rendered} ({how})"


def moved_cells(golden: str, rendered: str) -> list[Move]:
    """Every cell that differs. A line-count mismatch is one TEXT move on the
    first line past the shorter printout."""
    a, b = golden.split("\n"), rendered.split("\n")
    moves: list[Move] = []
    for i, (x, y) in enumerate(zip(a, b), start=1):
        if x == y:
            continue
        tx, ty = x.split(), y.split()
        if len(tx) != len(ty):
            moves.append(Move(i, -1, x.strip(), y.strip(), None))
            continue
        for j, (p, q) in enumerate(zip(tx, ty)):
            if p == q:
                continue
            vp, vq = number(p), number(q)
            if vp is None or vq is None:
                moves.append(Move(i, j, p, q, None))
                continue
            unit = max(last_digit(p), last_digit(q))
            moves.append(Move(i, j, p, q, abs(vp - vq) / unit))
    if len(a) != len(b):
        n = min(len(a), len(b))
        moves.append(Move(n + 1, -1, f"<{len(a)} lines>", f"<{len(b)} lines>", None))
    return moves
