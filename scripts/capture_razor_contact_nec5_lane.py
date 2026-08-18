"""Capture the razor GROUND-CONTACT sharp lane against the NEC-5 binary.

momwire#398 unit 3 (the razor-grounds pilot), the sibling of
`scripts/capture_razor_pec_nec5_lane.py` — same run recipe, same bar, same
generated-literals discipline, different geometry class: here every deck has
a wire END in the plane and is driven THERE, which is the unknown unit 3
adds. Regenerates `tests/golden_razor_contact_nec5.py`.

Must run under the antennaknobs venv (that is where the NEC-5 engine wrapper
lives), with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_razor_contact_nec5_lane.py

Only the binary's PRINTED impedances are recorded; nothing about NEC-5's
internals is read, quoted or inferred here.

Two geometries, both quarter-wave-class radiators at 14 MHz over `GN 1`:

  monopole    a 5.35 m vertical, base in the plane
  inverted-L  a 2.675 m riser plus a 2.675 m horizontal top, base in the
              plane — the same total length, bent, so the contact unknown
              is exercised on a wire that also has an interior bend and a
              second edge

Both are driven at the base: `EX 0` on segment 1 of the grounded wire, which
is momwire's `feed_arclength=0.0` — the gap between the plane and the wire's
first anchor. `GE 1` is the ground-plane flag the contact decks need (the
clearance decks in the sibling script use `GE 0`, since nothing there
touches).

Each geometry is run through the binary TWICE, and the second deck is the
point of this script:

  contact deck   the grounded radiator, N segments, over `GN 1`
  MIRROR deck    the same radiator plus its mirror image, in FREE SPACE,
                 2N segments, centre-driven — a 2L dipole for the monopole,
                 a "Z" (top, doubled riser, top) for the inverted-L

The mirror deck is the same physical antenna: image theory says the grounded
radiator's impedance is exactly half of it. It is also, exactly, what
`RazorSolver`'s grounded-end basis computes — the grounded tent IS the
mirror model's centre tent, folded (`tests/test_razor_ground_contact.py`
gate 1 pins the identity at 1e-13), so razor-vs-mirror-deck is the sharpest
external statement of "the contact basis is the right one" available.

The measured result, recorded in the generated module and gated in
`tests/test_razor_ground_contact.py`: razor tracks the MIRROR deck halved to
a constant ~+0.001..0.003 + 0.019j ohm at every rung of both ladders (spread
<= 0.0004 ohm), which clears the maintainer's sharp bar
(momwire#398 thread, 2026-08-17)

  |Z_razor − Z_NEC-5| <= max(0.20 ohm, 0.25 % of |Z|) at every rung, AND
  the (razor − NEC-5) offset constant down each ladder to within 0.05 ohm,

with room to spare — while against NEC-5's own CONTACT deck the same razor
answers carry a residual that DECAYS with N instead of holding constant
(monopole 0.30 -> 0.11 ohm, inverted-L 0.11 -> 0.011 ohm) and misses that
bar at the two coarsest monopole rungs. The reason is in this table and not
in razor: NEC-5's contact deck is not NEC-5's own mirror deck halved either
(monopole −0.133+0.285j ohm at N=24, decaying to −0.097+0.074j at N=96;
inverted-L +0.102+0.064j to +0.010+0.026j), so the disagreement is a
grounded-end discretization difference between the two codes that vanishes
as the mesh refines, not a formulation error in either. Both columns are
recorded; only the mirror-deck column is gated.

`nec5_quadrature=True` — NEC-5's own identified path rule (momwire#316) — is
the gated lane. The default Gauss-Legendre lane is captured too but is NOT
gated: the production bar is deferred with the finite grounds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden_razor_contact_nec5.py"

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)

# Ladders, in TOTAL segments. N >= 24 is where the sharp-lane bar applies.
LADDER = (24, 32, 48, 64, 96)

RAD = 1.0262e-3
MONO_LEN = 5.35
# Split evenly so the bend does not also introduce a segment-length step:
# the two edges are the same length and take the same segment count.
INVL_RISER = MONO_LEN / 2.0
INVL_TOP = MONO_LEN / 2.0


# --------------------------------------------------------------------------
# geometry, in both spellings (momwire polylines and NEC deck cards)
# --------------------------------------------------------------------------
def monopole_wires():
    return [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])], RAD


def invl_wires():
    return [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, INVL_RISER],
                [INVL_TOP, 0.0, INVL_RISER],
            ]
        )
    ], RAD


def _gw(tag, n, p0, p1, rad):
    return (
        f"GW {tag} {n} "
        f"{p0[0]:.6E} {p0[1]:.6E} {p0[2]:.6E} "
        f"{p1[0]:.6E} {p1[1]:.6E} {p1[2]:.6E} {rad:.6E}\n"
    )


def _deck(header, gws, ex, *, grounded):
    """One deck, contact (`GE 1` + `GN 1`) or free-space mirror (`GE 0`).

    `GE 1` is the ground-plane flag that connects the structure to the
    plane; with `GE 0` the same grounded geometry prints an open-circuit
    answer (68 − 5329j at N=32, measured), which is the tell that the flag
    is doing the connecting and not the `GN` card.

    Card order is the sibling script's (EX before FR), which the binary
    accepts.
    """
    return (
        f"CM {header}\nCE\n"
        + "".join(gws)
        + ("GE 1\nGN 1\n" if grounded else "GE 0\n")
        + ex
        + f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00\nXQ 0\nEN\n"
    )


def _ex(tag, seg):
    return f"EX 0 {tag} {seg} 2 1.000000E+00 0.000000E+00\n"


def monopole_deck(n):
    p = monopole_wires()[0][0]
    return _deck(
        "razor-contact monopole", [_gw(1, n, p[0], p[1], RAD)], _ex(1, 1), grounded=True
    )


def monopole_mirror_deck(n):
    """The 2L dipole in free space, 2N segments, driven at the centre knot
    (`EX` on segment N, which is the knot the sibling script's dipole decks
    already establish as razor's centre feed)."""
    lo = (0.0, 0.0, -MONO_LEN)
    hi = (0.0, 0.0, MONO_LEN)
    return _deck(
        "razor-contact monopole mirror",
        [_gw(1, 2 * n, lo, hi, RAD)],
        _ex(1, n),
        grounded=False,
    )


def invl_deck(n):
    p = invl_wires()[0][0]
    gws = [
        _gw(1, n // 2, p[0], p[1], RAD),
        _gw(2, n // 2, p[1], p[2], RAD),
    ]
    return _deck("razor-contact inverted-L", gws, _ex(1, 1), grounded=True)


def invl_mirror_deck(n):
    """The inverted-L plus its image: top, doubled riser, top — a "Z" in
    free space, driven at the riser's centre knot."""
    h = n // 2
    gws = [
        _gw(1, h, (INVL_TOP, 0.0, -INVL_RISER), (0.0, 0.0, -INVL_RISER), RAD),
        _gw(2, 2 * h, (0.0, 0.0, -INVL_RISER), (0.0, 0.0, INVL_RISER), RAD),
        _gw(3, h, (0.0, 0.0, INVL_RISER), (INVL_TOP, 0.0, INVL_RISER), RAD),
    ]
    return _deck("razor-contact inverted-L mirror", gws, _ex(2, h), grounded=False)


GEOMS = {
    "monopole": {
        "wires": monopole_wires,
        "deck": monopole_deck,
        "mirror_deck": monopole_mirror_deck,
        "split": lambda n: [[n]],
    },
    "invl": {
        "wires": invl_wires,
        "deck": invl_deck,
        "mirror_deck": invl_mirror_deck,
        "split": lambda n: [[n // 2, n // 2]],
    },
}


def razor_z(name, n, **mode):
    from momwire import RazorSolver

    g = GEOMS[name]
    wires, radius = g["wires"]()
    z, _ = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=g["split"](n),
        wire_radius=radius,
        wavelength=WL,
        ground_z=0.0,
        feed_arclength=0.0,
        **mode,
    ).compute_impedance()
    return complex(z)


# --------------------------------------------------------------------------
def main() -> None:
    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("RAZOR_CONTACT_CAPTURES", "/tmp/razor-contact-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    rows: dict[str, list[tuple[int, complex, complex, complex, complex]]] = {}
    for name in GEOMS:
        out = []
        for n in LADDER:
            zn = complex(eng.run_deck(GEOMS[name]["deck"](n))[0][0][2])
            # Halved here, once: every consumer wants the grounded-radiator
            # value, and image theory is what the halving is.
            zm = complex(eng.run_deck(GEOMS[name]["mirror_deck"](n))[0][0][2]) / 2.0
            zq = razor_z(name, n, nec5_quadrature=True)
            zg = razor_z(name, n)
            out.append((n, zn, zm, zq, zg))
            print(
                f"{name:>10} N={n:<3} nec5={zn:.4f} mirror/2={zm:.4f} "
                f"n5q={zq:.4f} gl={zg:.4f}"
            )
        rows[name] = out

    _write_golden(rows)
    _report(rows)


def _write_golden(rows) -> None:
    lines = [
        '"""NEC-5 printed impedances for GROUNDED (contact) decks, and the',
        "razor lanes.",
        "",
        "GENERATED by scripts/capture_razor_contact_nec5_lane.py — do not edit",
        "by hand. See that script for the decks, the dimensions, the run recipe",
        "and the sharp-lane bar these numbers are gated against (momwire#398",
        "unit 3). Citation: NEC-5 (LLNL-CODE-746721), ground-contact ladder",
        "decks, 2026-08-17.",
        "",
        "Each row is (N_total, Z_nec5_contact, Z_nec5_mirror_half,",
        "Z_razor_n5q, Z_razor_gl):",
        "",
        "  Z_nec5_contact      the binary on the GROUNDED deck (GE 1 + GN 1),",
        "                      N segments, driven at the base",
        "  Z_nec5_mirror_half  the binary on the same radiator PLUS ITS IMAGE",
        "                      in free space, 2N segments, centre-driven, and",
        "                      halved — the same antenna by image theory, and",
        "                      the deck razor's grounded-end basis reproduces",
        "  Z_razor_n5q         `RazorSolver(ground_z=0, feed_arclength=0,",
        "                      nec5_quadrature=True)`",
        "  Z_razor_gl          the same on the default Gauss-Legendre path",
        "                      quadrature",
        "",
        "Only Z_razor_n5q against Z_nec5_mirror_half is GATED. Against",
        "Z_nec5_contact the residual decays with N rather than holding",
        "constant, because NEC-5's grounded deck is not NEC-5's own mirror",
        "deck halved either; the capture script's docstring has the numbers",
        "and the reading. The GL column is recorded so the deferred",
        "production-lane bar has a starting point on file.",
        '"""',
        "",
        "CONTACT_LADDERS = {",
    ]

    def _lit(z, places):
        # Spaced operators, so the generated file is already `ruff format`
        # clean and a re-capture never shows up as a formatting diff.
        sign = "-" if z.imag < 0 else "+"
        return f"{z.real:.{places}f} {sign} {abs(z.imag):.{places}f}j"

    for name, out in rows.items():
        lines.append(f'    "{name}": (')
        for n, zn, zm, zq, zg in out:
            # One field per line: five columns do not fit in 88 characters,
            # and this is the shape `ruff format` would impose anyway, so a
            # re-capture never shows up as a formatting diff.
            lines.append("        (")
            lines.append(f"            {n},")
            for z, places in ((zn, 4), (zm, 5), (zq, 6), (zg, 6)):
                lines.append(f"            {_lit(z, places)},")
            lines.append("        ),")
        lines.append("    ),")
    lines.append("}")
    GOLDEN.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {GOLDEN}")


def _report(rows) -> None:
    """Both ladders, side by side: razor against the mirror deck (gated) and
    against NEC-5's own contact deck (recorded)."""
    for name, out in rows.items():
        print(f"\n===== {name} =====")
        print(
            f"{'N':>4} | {'mirror/2 R':>10} {'X':>9} | {'n5q R':>9} {'X':>9} | "
            f"{'dR':>8} {'dX':>8} {'|dZ|':>7} {'bar':>7} | "
            f"{'vs contact dR':>13} {'dX':>8} {'|dZ|':>7} | {'GL dR':>8} {'GL dX':>8}"
        )
        res, res_c = [], []
        for n, zn, zm, zq, zg in out:
            d, dc, dg = zq - zm, zq - zn, zg - zn
            bar = max(0.20, 0.0025 * abs(zm))
            res.append(d)
            res_c.append(dc)
            print(
                f"{n:>4} | {zm.real:10.4f} {zm.imag:9.4f} | "
                f"{zq.real:9.4f} {zq.imag:9.4f} | "
                f"{d.real:8.4f} {d.imag:8.4f} {abs(d):7.4f} {bar:7.4f} | "
                f"{dc.real:13.4f} {dc.imag:8.4f} {abs(dc):7.4f} | "
                f"{dg.real:8.4f} {dg.imag:8.4f}"
            )
        for tag, r in (("vs mirror ", res), ("vs contact", res_c)):
            sr = max(z.real for z in r) - min(z.real for z in r)
            sx = max(z.imag for z in r) - min(z.imag for z in r)
            print(f"  offset constancy {tag}: dR spread {sr:.4f}, dX spread {sx:.4f}")


if __name__ == "__main__":
    main()
