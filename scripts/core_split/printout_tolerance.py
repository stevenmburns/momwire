"""Two printouts compared within a tolerance, for the Windows gates (momwire#1558).

Windows needs no bit-compatibility (Steve, 2026-10-03): the MSVC build is
/fp:fast, so where the C++ route answers a deck the printout can differ from
Python's in the digits that cancellation leaves -- the imaginary part of a
1e10-ohm open load printed as 7.1780E-07 by one and 1.4356E-06 by the other,
next to a real part of 1.0000E+10 that both print the same. Byte equality
cannot hold there, and a relative tolerance on each number alone would call
those 100 % apart. What both are is the same multiple of eps of the numbers
they were computed WITH.

So a printout is compared line by line: the text between the numbers must be
the same (whitespace runs aside, which a sign can shift), the same count of
numbers in each line, and each pair of numbers must agree

    |got - ref|  <=  rtol * S,   S = max |x| over that token and its two
                                     neighbours on the line, both printouts

or differ by exactly one unit of the last printed digit (the same value
rounded either side of a printed boundary). :func:`ratio` measures the worst
|got - ref| / S over a printout; the gate's rtol is DERIVED from it
(exe_compare.py, PRINTOUT_RTOL_WIN32).
"""

from __future__ import annotations

import math
import re

# A printed number, INF and NAN included: a value one side prints finite and
# the other INF is a numeric difference (an infinite one), not a text one.
_NUMBER = re.compile(
    rb"[-+]?(?:(?<![A-Za-z])(?:INF|NAN)(?![A-Za-z])|(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)"
)
_SPACE = re.compile(rb"\s+")


def _unit(token: bytes) -> float:
    """One unit of the token's last printed digit; 0.0 where that says
    nothing: INF, NAN, and a zero in E format (0.0000E+00 is printed with
    exponent 0, so its "last digit" would be 1e-4 whatever it stands beside)."""
    t = token.lstrip(b"+-").upper().replace(b"D", b"E")
    if t in (b"INF", b"NAN"):
        return 0.0
    mant, _, exp = t.partition(b"E")
    if exp and float(mant) == 0.0:
        return 0.0
    decimals = len(mant.partition(b".")[2])
    return 10.0 ** ((int(exp) if exp else 0) - decimals)


def _split(line: bytes):
    numbers = [m.group() for m in _NUMBER.finditer(line)]
    text = [_SPACE.sub(b" ", part).strip() for part in _NUMBER.split(line)]
    return text, numbers


def ratio(ref: bytes, got: bytes) -> float | None:
    """The worst |got - ref| / S over two printouts (0.0 when every number
    agrees or differs only by a printed-digit flip); None when they differ in
    anything but numbers: a line, the text between numbers, a number count."""
    rl, gl = ref.split(b"\r\n"), got.split(b"\r\n")
    if len(rl) != len(gl):
        return None
    worst = 0.0
    for r, g in zip(rl, gl, strict=True):
        if r == g:
            continue
        rt, rn = _split(r)
        gt, gn = _split(g)
        if rt != gt or len(rn) != len(gn):
            return None
        rv = [float(x.upper().replace(b"D", b"E")) for x in rn]
        gv = [float(x.upper().replace(b"D", b"E")) for x in gn]
        for k, (a, b) in enumerate(zip(rv, gv, strict=True)):
            if a == b or (math.isnan(a) and math.isnan(b)):
                continue
            if not (math.isfinite(a) and math.isfinite(b)):
                return math.inf  # INF or NAN on one side only
            d = abs(a - b)
            if d <= 1.000001 * min(_unit(rn[k]), _unit(gn[k])):
                continue  # one printed digit: a rounding boundary
            lo, hi = max(0, k - 1), min(len(rv), k + 2)
            near = [abs(x) for x in rv[lo:hi] + gv[lo:hi] if math.isfinite(x)]
            s = max(near)
            worst = max(worst, d / s if s else math.inf)
    return worst
