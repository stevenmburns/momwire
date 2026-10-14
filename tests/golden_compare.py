"""Reading two printouts of one deck cell by cell (python-free arc, step 0).

`undefined_blanked` is the boundary of what the golden corpus claims across
machines: momwire#578's four rules (`eznec_reproducibility.blank_undefined`)
plus a fifth, measured here, for the same cancellation crumb in a network
row's VOLTAGE and CURRENT cells.

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

import eznec_reproducibility as R

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


# Rule 5. A network row's voltage or current that a source or a gyrator pins
# real (or imaginary) prints the other part as the crumb of a cancellation:
# `1.4142E+00 -3.5128E-17`, whose sign and size depend on the order a
# reduction ran in. Measured over 910 printouts (this corpus, and both bases
# over the 401-deck step-0 census), |small part| / |large part| of every V and
# I cell pair:
#
#     1e-18 .. 1e-17     6 pairs    the gyrator decks: crumbs
#     1e-16 .. 1e-12     0
#     1e-11 and up    2026 pairs    readings (the weakest 3.4e-11, a load probe)
#
# A Linux CI runner moved exactly those six cells against Haswell. The bar
# sits mid-void on a log scale, three decades from each side.
VI_CRUMB = 1e-14
_V_RE, _V_IM = 0, 1


def _vi_crumbs(line: str) -> str:
    cells = R._cells(line)
    v = [R._f(c) for c in cells]
    if any(x is None for x in v[:4]):
        return line
    blank = []
    for re_, im in ((_V_RE, _V_IM), (R._I_RE, R._I_IM)):
        big = max(abs(v[re_]), abs(v[im]))
        for k in (re_, im):
            if big and abs(v[k]) < VI_CRUMB * big:
                blank.append(k)
    if not blank:
        return line
    R._blank(cells, tuple(blank))
    return line[: R._CELL0] + "".join(cells)


def undefined_blanked(text: str) -> str:
    """``text`` with every cell the arithmetic does not define replaced by a
    marker: momwire#578's rules, then rule 5 on the network rows. Applied to
    both sides of a comparison, so it can only narrow what is asserted."""
    lines = R.blank_undefined(text).split("\n")
    for i, line in enumerate(lines):
        if R._NETWORK_HEAD not in line:
            continue
        j = i + R._SKIP[R._NETWORK_HEAD] + 1
        while j < len(lines) and lines[j].strip():
            lines[j] = _vi_crumbs(lines[j])
            j += 1
    return "\n".join(lines)


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
            # A printed zero has no last digit of its own: measure against
            # the nonzero side's.
            units = [last_digit(t) for t, v in ((p, vp), (q, vq)) if v != 0]
            unit = max(units) if units else last_digit(p)
            moves.append(Move(i, j, p, q, abs(vp - vq) / unit))
    if len(a) != len(b):
        n = min(len(a), len(b))
        moves.append(Move(n + 1, -1, f"<{len(a)} lines>", f"<{len(b)} lines>", None))
    return moves
