"""Capture the razor PEC sharp lane against the licensed NEC-5 binary.

momwire#398 unit 2 (the razor-grounds pilot). Runs the four ladder
geometries over a perfectly conducting ground — the free-space agreement-
floor decks lifted clear of the plane, plus a `GN 1` card — through the
NEC-5 binary and through `RazorSolver(ground_z=0.0)` in both quadrature
lanes, then regenerates `tests/golden_razor_pec_nec5.py` as pure literals so
the momwire test suite needs no binary and no antennaknobs dependency.

Must run under the antennaknobs venv (that is where the NEC-5 engine wrapper
lives), with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_razor_pec_nec5_lane.py

Only the binary's PRINTED impedances are recorded; nothing about NEC-5's
internals is read, quoted or inferred here.

The geometries are the free-space floor study's, verbatim except for one
change per deck: every conductor is raised so it stands clear of `z = 0`.
That clearance is required, not cosmetic — unit 2 ships the PEC FILL, and
ground CONTACT (a wire end in the plane, whose current the image has to
continue) is refused by `RazorSolver`. Heights:

  dipole / fat-dipole   horizontal at z = 5 m           (0.233 wavelength)
  inverted-V            apex at z = 10 m, ends at 6.22 m
  loop                  horizontal, in the z = 5 m plane

The rungs start at N = 24 total, which is where the maintainer's approved
sharp-lane bar applies (momwire#398 thread, 2026-08-17):

  |Z_razor − Z_NEC-5| <= max(0.20 ohm, 0.25 % of |Z|) at every rung, AND
  the (razor − NEC-5) offset constant down each ladder to within 0.05 ohm,

with `nec5_quadrature=True` — NEC-5's own identified path rule (momwire#316)
— as the gated lane. The default Gauss-Legendre lane is captured too but is
NOT gated: the production bar is deferred with the finite grounds.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden_razor_pec_nec5.py"

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)

# Ladders, in TOTAL segments. N >= 24 is where the sharp-lane bar applies.
LADDER = (24, 32, 48, 64, 96)

DIP_LEN = 10.18946
DIP_RAD = 1.0262e-3
FAT_RAD = 1.0e-2
DIP_H = 5.0

INVVEE_ARM = 5.35
INVVEE_C = INVVEE_ARM / math.sqrt(2.0)
INVVEE_APEX_H = 10.0

LOOP_SIDE = 6.05
LOOP_H = 5.0
LOOP_RAD = 2.0e-3


# --------------------------------------------------------------------------
# geometry, in both spellings (momwire polylines and NEC deck cards)
# --------------------------------------------------------------------------
def dipole_wires(radius):
    pl = np.array([[0.0, 0.0, DIP_H], [0.0, DIP_LEN, DIP_H]])
    return [pl], radius, [None]


def invvee_wires():
    a = INVVEE_APEX_H
    pl = np.array(
        [
            [-INVVEE_C, 0.0, a - INVVEE_C],
            [0.0, 0.0, a],
            [INVVEE_C, 0.0, a - INVVEE_C],
        ]
    )
    return [pl], 1.0e-3, [None]


def loop_wires():
    s, h = LOOP_SIDE, LOOP_H
    pl = np.array(
        [
            [0.0, 0.0, h],
            [s, 0.0, h],
            [s, s, h],
            [0.0, s, h],
            [0.0, 0.0, h],
        ]
    )
    return [pl], LOOP_RAD, [None]


def _gw(tag, n, p0, p1, rad):
    return (
        f"GW {tag} {n} "
        f"{p0[0]:.6E} {p0[1]:.6E} {p0[2]:.6E} "
        f"{p1[0]:.6E} {p1[1]:.6E} {p1[2]:.6E} {rad:.6E}\n"
    )


def _deck(header, gws, ex_tag, ex_seg):
    """The floor study's deck, plus one `GN 1` card after `GE`.

    Card order is the floor decks' unchanged (EX before FR), which the
    binary accepts; `GE 0` / `GE -1` / `GE 1` were all measured to give the
    same printed impedance here, since nothing touches the plane.
    """
    return (
        f"CM {header}\nCE\n"
        + "".join(gws)
        + "GE 0\n"
        + "GN 1\n"
        + f"EX 0 {ex_tag} {ex_seg} 2 1.000000E+00 0.000000E+00\n"
        + f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00\nXQ 0\nEN\n"
    )


def dipole_deck(n, radius, note):
    p = dipole_wires(radius)[0][0]
    return _deck(f"razor-pec {note}", [_gw(1, n, p[0], p[1], radius)], 1, n // 2)


def invvee_deck(n):
    a = INVVEE_APEX_H
    lo = (-INVVEE_C, 0.0, a - INVVEE_C)
    hi = (INVVEE_C, 0.0, a - INVVEE_C)
    apex = (0.0, 0.0, a)
    gws = [
        _gw(1, n // 2, lo, apex, 1.0e-3),
        _gw(2, n // 2, apex, hi, 1.0e-3),
    ]
    return _deck("razor-pec invvee", gws, 1, n // 2)


def loop_deck(n):
    s, h, n_side = LOOP_SIDE, LOOP_H, n // 4
    corners = [(0.0, 0.0, h), (s, 0.0, h), (s, s, h), (0.0, s, h)]
    gws = [
        _gw(t + 1, n_side, corners[t], corners[(t + 1) % 4], LOOP_RAD) for t in range(4)
    ]
    return _deck("razor-pec loop", gws, 1, n_side // 2)


GEOMS = {
    "dipole": {
        "wires": lambda: dipole_wires(DIP_RAD),
        "deck": lambda n: dipole_deck(n, DIP_RAD, "dipole"),
        "split": lambda n: [[n]],
        "feed": {},
    },
    "fat-dipole": {
        "wires": lambda: dipole_wires(FAT_RAD),
        "deck": lambda n: dipole_deck(n, FAT_RAD, "fat"),
        "split": lambda n: [[n]],
        "feed": {},
    },
    "invvee": {
        "wires": invvee_wires,
        "deck": invvee_deck,
        "split": lambda n: [[n // 2, n // 2]],
        "feed": {},
    },
    "loop": {
        "wires": loop_wires,
        "deck": loop_deck,
        "split": lambda n: [[n // 4] * 4],
        "feed": {"feed_arclength": LOOP_SIDE / 2},
    },
}


def razor_z(name, n, **mode):
    from momwire import RazorSolver

    g = GEOMS[name]
    wires, radius, _ = g["wires"]()
    z, _ = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=g["split"](n),
        wire_radius=radius,
        wavelength=WL,
        ground_z=0.0,
        **g["feed"],
        **mode,
    ).compute_impedance()
    return complex(z)


# --------------------------------------------------------------------------
def main() -> None:
    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(os.environ.get("RAZOR_PEC_CAPTURES", "/tmp/razor-pec-captures"))
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    rows: dict[str, list[tuple[int, complex, complex, complex]]] = {}
    for name in GEOMS:
        out = []
        for n in LADDER:
            zn = complex(eng.run_deck(GEOMS[name]["deck"](n))[0][0][2])
            zq = razor_z(name, n, nec5_quadrature=True)
            zg = razor_z(name, n)
            out.append((n, zn, zq, zg))
            print(f"{name:>10} N={n:<3} nec5={zn:.4f} n5q={zq:.4f} gl={zg:.4f}")
        rows[name] = out

    _write_golden(rows)
    _report(rows)


def _write_golden(rows) -> None:
    lines = [
        '"""NEC-5 printed impedances over a PEC ground, and the razor lanes.',
        "",
        "GENERATED by scripts/capture_razor_pec_nec5_lane.py — do not edit by",
        "hand. See that script for the decks, the heights, the run recipe and",
        "the sharp-lane bar these numbers are gated against (momwire#398 unit",
        "2). Citation: NEC-5 (LLNL-CODE-746721), PEC-ground ladder decks,",
        "2026-08-17.",
        "",
        "Each row is (N_total, Z_nec5, Z_razor_n5q, Z_razor_gl): the binary's",
        "printed impedance, `RazorSolver(ground_z=0, nec5_quadrature=True)`,",
        "and `RazorSolver(ground_z=0)` on its default Gauss-Legendre path",
        "quadrature. Only the n5q column is gated (the GL / production bar is",
        "deferred with the finite grounds); the GL column is recorded so the",
        "deferred bar's starting point is on the record.",
        '"""',
        "",
        "PEC_LADDERS = {",
    ]

    def _lit(z, places):
        # Spaced operators, so the generated file is already `ruff format`
        # clean and a re-capture never shows up as a formatting diff.
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
        print(f"\n===== {name} (PEC, GN 1) =====")
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
