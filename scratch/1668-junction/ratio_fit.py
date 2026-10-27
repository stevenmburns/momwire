"""The momwire#1668 diagnostic: is a junction residual the members' charge
RATIO alone?

Scale one member's junction factor by lam (the others at the node kept) and
solve lam, together with a floor F common to both decks, so that two decks
of one geometry fed at two positions both land on NEC-4.2:

    X_momwire(deck, lam) - X_nec42(deck) = F      for deck in (A, B)

If the residual is the ratio alone, the same lam serves every feed position
(the miss scales with how much charge the feed puts at the node, but the
ratio that cancels it does not move) and F comes out as the uniform-wire
floor of that mesh. Both held on every family in `results/`, under the
#1644 solve, to 1e-4 of lam; they are what pinned the readings
`_nec4_junction` now takes. Run against today's solve the same fit returns
lam = 1 (to the floor's noise).

    python scratch/1668-junction/ratio_fit.py DIR DECK_A DECK_B SELECT [--1644]

DIR holds DECK.nec and NEC-4.2's DECK.out (`run_nec42.sh DIR`). SELECT
picks the scaled member(s): a radius in metres, `perp` (direction along z),
`negz` (along -z) or `notx` (not along x). `--1644` solves with the n = 2
node-read solve #1644 shipped, kept below for that purpose.
"""

from __future__ import annotations

import math
import re
import sys
import warnings
from pathlib import Path

from momwire import _nec4_junction as nj
from momwire.eznec import _nec4

_NUM = re.compile(r"[-+]?\d\.\d+E[-+]\d\d")


def nec42_z(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" in line:
            v = _NUM.findall(lines[i + 4])
            return complex(float(v[4]), float(v[5]))
    raise ValueError(f"{path}: no ANTENNA INPUT PARAMETERS")


def densities_1644(k, dirs, radii, lengths):
    """#1644's solve: n = 2, each member's own length, read at the node."""
    m = len(radii)
    mat = []
    for i in range(m):
        for s in (0.5 * lengths[i], lengths[i]):
            p = [s * c for c in dirs[i]]
            row = []
            for ell in range(m):
                u = dirs[ell]
                z = sum(pc * uc for pc, uc in zip(p, u, strict=True))
                x = [pc - z * uc for pc, uc in zip(p, u, strict=True)]
                b = math.sqrt(radii[ell] ** 2 + sum(v * v for v in x))
                row.append(nj._triangle(k, b, z, lengths[ell]))
                row.append(lengths[ell] * nj._semi_infinite(k, b, z))
            mat.append(row)
    q = nj._solve(mat, [1.0] * (2 * m))
    return [(q[2 * ell] + q[2 * ell + 1]) * lengths[ell] for ell in range(m)]


def selector(spec):
    if spec == "perp":
        return lambda d, r: abs(d[2]) > 0.5
    if spec == "negz":
        return lambda d, r: d[2] < -0.99
    if spec == "notx":
        return lambda d, r: abs(d[0]) < 0.5
    radius = float(spec)
    return lambda d, r: abs(r - radius) < 1e-12


def solve_x(text, lam, member, old):
    base = densities_1644 if old else nj.junction_densities

    def factors(k, dirs, radii, lengths, first):
        dens = base(k, dirs, radii, lengths)
        out = [first * q / dens[0] for q in dens]
        return [f * lam if member(dirs[i], radii[i]) else f for i, f in enumerate(out)]

    keep = nj.junction_charge_factors
    nj.junction_charge_factors = factors
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _nec4.solve(text, basis="sinusoidal").runs[0].sources[0].impedance
    finally:
        nj.junction_charge_factors = keep


def main(argv):
    old = "--1644" in argv
    d, a, b, spec = (x for x in argv if x != "--1644")
    member = selector(spec)
    res = {}
    for name in (a, b):
        text = (Path(d) / f"{name}.nec").read_text()
        ref = nec42_z(Path(d) / f"{name}.out")
        res[name] = [(solve_x(text, lam, member, old) - ref).imag for lam in (1, 1.01)]
    (a0, a1), (b0, b1) = res[a], res[b]
    sa, sb = (a1 - a0) / 0.01, (b1 - b0) / 0.01
    lam = 1 + (b0 - a0) / (sa - sb)
    floor = a0 + sa * (lam - 1)
    print(f"{a} {b}: lam* {lam:.5f} floor {floor:+.3f}  (raw dX {a0:+.3f} {b0:+.3f})")


if __name__ == "__main__":
    main(sys.argv[1:])
