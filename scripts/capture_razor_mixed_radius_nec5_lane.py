"""Capture the razor MIXED-RADIUS sharp lane against the licensed NEC-5 binary.

momwire#147 on `RazorSolver` — per-wire radii, the last cell of the
capability row this formulation could fill without a new physics object.
Runs two mixed-radius geometries in free space and over a `GN 1` plane
through the NEC-5 binary and through `RazorSolver(wire_radius=[…])` in both
quadrature lanes, then regenerates `tests/golden_razor_mixed_radius_nec5.py`
as pure literals so the momwire suite needs no binary and no antennaknobs
dependency.

Must run under the antennaknobs venv (that is where the NEC-5 engine wrapper
lives), with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_razor_mixed_radius_nec5_lane.py

Only the binary's PRINTED impedances are recorded; nothing about NEC-5's
internals is read, quoted or inferred here.

**Why a NEC deck is a like-for-like oracle for this capability.** A `GW`
card carries its own radius, so a mixed-radius model is native on both
sides — there is no translation step in which a convention could hide. Two
geometries, because they exercise the two regimes a per-wire radius has:

  `parasitic`  a thin driven dipole beside a FAT parasitic reflector,
               0.15 lambda apart — different radii on wires that never
               touch, the ordinary Yagi/quad case;
  `stepped`    one dipole whose inner half is fat and whose outer half is
               thin, meeting AT the fed knot — different radii at a
               JUNCTION, the case where the reduced kernel's a² is the
               whole of the perpendicular distance and the source-radius
               convention (`RazorSolver._seg_moments_prepare`) is
               therefore under test rather than along for the ride.

Both are captured in free space and raised to z = 5 m over a `GN 1` plane
(nothing touches it — this formulation refuses ground CONTACT over anything
but PEC, and this lane is PEC).

The bar is momwire#398 unit 2's, unchanged:

  |Z_razor − Z_NEC-5| <= max(0.20 ohm, 0.25 % of |Z|) at every rung, AND
  the (razor − NEC-5) offset constant down each ladder to within 0.05 ohm,

with `nec5_quadrature=True` as the gated lane. The default Gauss-Legendre
lane is captured too but is not gated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden_razor_mixed_radius_nec5.py"

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)

# Ladders, in TOTAL segments. N >= 24 is where the sharp-lane bar applies.
LADDER = (24, 32, 48, 64, 96)

THIN = 1.0262e-3
FAT = 1.0e-2
DIP_LEN = 10.18946
REFL_LEN = 10.70
SPACING = 3.21  # 0.15 lambda at 14 MHz
HI = 5.0


# --------------------------------------------------------------------------
# geometry, in both spellings (momwire polylines and NEC deck cards)
# --------------------------------------------------------------------------
def _gw(tag, n, p0, p1, rad):
    return (
        f"GW {tag} {n} "
        f"{p0[0]:.6E} {p0[1]:.6E} {p0[2]:.6E} "
        f"{p1[0]:.6E} {p1[1]:.6E} {p1[2]:.6E} {rad:.6E}\n"
    )


def _deck(header, gws, ex_tag, ex_seg, grounded):
    return (
        f"CM {header}\nCE\n"
        + "".join(gws)
        + "GE 0\n"
        + ("GN 1\n" if grounded else "")
        + f"EX 0 {ex_tag} {ex_seg} 2 1.000000E+00 0.000000E+00\n"
        + f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00\nXQ 0\nEN\n"
    )


def parasitic(n, h):
    """Thin driven dipole + fat parasitic reflector, 0.15 lambda apart."""
    d0 = (0.0, -DIP_LEN / 2, h)
    d1 = (0.0, DIP_LEN / 2, h)
    r0 = (SPACING, -REFL_LEN / 2, h)
    r1 = (SPACING, REFL_LEN / 2, h)
    wires = [np.array([list(d0), list(d1)]), np.array([list(r0), list(r1)])]
    deck = _deck(
        "razor mixed parasitic",
        [_gw(1, n // 2, d0, d1, THIN), _gw(2, n // 2, r0, r1, FAT)],
        1,
        n // 4,
        h > 0.0,
    )
    return wires, [THIN, FAT], [[n // 2], [n // 2]], deck, {}


def stepped(n, h):
    """One dipole: fat inner half meeting thin outer half AT the fed knot."""
    a0 = (0.0, -DIP_LEN / 2, h)
    mid = (0.0, 0.0, h)
    a1 = (0.0, DIP_LEN / 2, h)
    wires = [np.array([list(a0), list(mid)]), np.array([list(mid), list(a1)])]
    deck = _deck(
        "razor mixed stepped",
        [_gw(1, n // 2, a0, mid, FAT), _gw(2, n // 2, mid, a1, THIN)],
        1,
        n // 2,
        h > 0.0,
    )
    feed = dict(feed_wire_index=0, feed_arclength=DIP_LEN / 2)
    return wires, [FAT, THIN], [[n // 2], [n // 2]], deck, feed


GEOMS = {"parasitic": parasitic, "stepped": stepped}
HEIGHTS = {"free": 0.0, "pec": HI}


def razor_z(builder, n, h, **mode):
    from momwire import RazorSolver

    wires, radii, split, _deck, feed = builder(n, h)
    z, _ = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=split,
        wire_radius=radii,
        wavelength=WL,
        ground_z=0.0 if h > 0.0 else None,
        **feed,
        **mode,
    ).compute_impedance()
    return complex(z)


# --------------------------------------------------------------------------
def main() -> None:
    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(os.environ.get("RAZOR_MIXED_CAPTURES", "/tmp/razor-mixed-captures"))
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    rows: dict[str, list[tuple[int, complex, complex, complex]]] = {}
    for name, builder in GEOMS.items():
        for gtag, h in HEIGHTS.items():
            out = []
            for n in LADDER:
                deck = builder(n, h)[3]
                zn = complex(eng.run_deck(deck)[0][0][2])
                zq = razor_z(builder, n, h, nec5_quadrature=True)
                zg = razor_z(builder, n, h)
                out.append((n, zn, zq, zg))
                print(
                    f"{name:>10}/{gtag:<4} N={n:<3} nec5={zn:.4f} "
                    f"n5q={zq:.4f} gl={zg:.4f}"
                )
            rows[f"{name}-{gtag}"] = out

    _write_golden(rows)
    _report(rows)


def _write_golden(rows) -> None:
    lines = [
        '"""NEC-5 printed impedances on MIXED-RADIUS decks, and the razor lanes.',
        "",
        "GENERATED by scripts/capture_razor_mixed_radius_nec5_lane.py — do not",
        "edit by hand. See that script for the decks, the heights, the run",
        "recipe and the sharp-lane bar these numbers are gated against",
        "(momwire#147 on RazorSolver). Citation: NEC-5 (LLNL-CODE-746721),",
        "mixed-radius ladder decks, 2026-08-18.",
        "",
        "Keys are `<geometry>-<ground>`; each row is",
        "(N_total, Z_nec5, Z_razor_n5q, Z_razor_gl). Only the n5q column is",
        "gated, exactly as in golden_razor_pec_nec5.py.",
        '"""',
        "",
        f"THIN = {THIN!r}",
        f"FAT = {FAT!r}",
        f"DIP_LEN = {DIP_LEN!r}",
        f"REFL_LEN = {REFL_LEN!r}",
        f"SPACING = {SPACING!r}",
        f"HI = {HI!r}",
        f"FREQ_MHZ = {FREQ_MHZ!r}",
        "",
        "MIXED_LADDERS = {",
    ]

    def _lit(z, places):
        sign = "-" if z.imag < 0 else "+"
        return f"{z.real:.{places}f} {sign} {abs(z.imag):.{places}f}j"

    for name, out in rows.items():
        lines.append(f'    "{name}": (')
        for n, zn, zq, zg in out:
            lines.append(f"        ({n}, {_lit(zn, 4)}, {_lit(zq, 6)}, {_lit(zg, 6)}),")
        lines.append("    ),")
    lines.append("}")
    GOLDEN.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {GOLDEN}")


def _report(rows) -> None:
    for name, out in rows.items():
        print(f"\n===== {name} =====")
        print(
            f"{'N':>4} | {'NEC-5 R':>9} {'X':>9} | {'n5q R':>9} {'X':>9} | "
            f"{'dR':>8} {'dX':>8} | {'|dZ|':>7} {'bar':>7} | "
            f"{'GL dR':>8} {'GL dX':>8}"
        )
        res = []
        for n, zn, zq, zg in out:
            d, dg = zq - zn, zg - zn
            bar = max(0.20, 0.0025 * abs(zn))
            res.append(d)
            print(
                f"{n:>4} | {zn.real:9.4f} {zn.imag:9.4f} | "
                f"{zq.real:9.4f} {zq.imag:9.4f} | "
                f"{d.real:8.4f} {d.imag:8.4f} | {abs(d):7.4f} {bar:7.4f} | "
                f"{dg.real:8.4f} {dg.imag:8.4f}"
            )
        spread_r = max(r.real for r in res) - min(r.real for r in res)
        spread_x = max(r.imag for r in res) - min(r.imag for r in res)
        print(f"  offset constancy: dR spread {spread_r:.4f}, dX spread {spread_x:.4f}")


if __name__ == "__main__":
    main()
