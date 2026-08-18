"""Capture the razor LOADING lane against the licensed NEC-5 binary.

momwire#427. Runs a free-space ByDipole1 ladder four ways — unloaded, a
lumped `LD 4` at the FED knot, the same lumped load at a mid-element knot,
and `LD 5` copper conductivity over the whole wire — through the NEC-5
binary and through `RazorSolver` in both quadrature lanes, then regenerates
`tests/golden_razor_loading_nec5.py` as pure literals so the momwire test
suite needs no binary and no antennaknobs dependency.

Must run under the antennaknobs venv (that is where the NEC-5 engine wrapper
lives), with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_razor_loading_nec5_lane.py

Only the binary's PRINTED impedances are recorded; nothing about NEC-5's
internals is read, quoted or inferred here.

The convention verification, done BEFORE anything is gated
--------------------------------------------------------
A twin claim is only worth making like-for-like, so this capture first
establishes what the binary's `LD` cards MEAN in terms this formulation can
say, using printed numbers alone (the unit-3 mirror-deck lesson).

* **Where an `LD` sits.** `LD 4 tag j j` on the fed segment moves the
  printed impedance by exactly the card's R + jX (probe at N=24: unloaded
  66.911 − 34.179j, loaded with R=50 gives 116.910 − 34.179j, i.e.
  ΔZ = 49.999 + 0.000j to print resolution). That is the drive-point
  identity, so `LD 4` is a series impedance in the port's own current path
  — the same object `lumped_loads` is here. And since `EX 0 tag j 2` feeds
  the KNOT at segment j's far end, an `LD` on segment j landing on the same
  site as an `EX` on segment j fixes the addressing: **`LD` on segment j is
  a lumped load at knot j**, which is one basis coefficient here, not a
  segment-wide distribution. The mid-element column below is the check that
  reads across the whole ladder rather than at one N.
* **What `LD 5` is.** A per-unit-length conductor impedance, the same axis
  `wire_conductivity` drives. At N=24 the binary's loading increment is
  0.818 + 0.721j Ω against momwire's 0.8189 + 0.7218j — agreement at the
  binary's print resolution (1e-3 Ω), which is finer than the ~0.6 %
  separation between the exact solid-cylinder I₀/I₁ internal impedance
  `_wire_loading.wire_internal_impedance` computes and the strong-skin
  asymptote. So the distributed convention is like-for-like too, and gets
  gated here rather than deferred to the cross-formulation oracle.

What is gated
-------------
The DIFFERENCE OF COLUMNS. Razor and NEC-5 do not agree pointwise at any
finite N — that is the twin's whole story (`tests/test_razor_nec5_twin.py`)
— so the loading claim is that loading adds NO new gap:

    | (Z_loaded − Z_unloaded)_razor − (Z_loaded − Z_unloaded)_NEC5 |

is held to the sharp bar at every rung, in the `nec5_quadrature` lane
(momwire#316's identified path rule). The absolute columns are recorded too
so the underlying twin gap stays visible and the increment claim can be read
against it. The default Gauss-Legendre lane is captured but not gated, the
same way `scripts/capture_razor_pec_nec5_lane.py` leaves it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden_razor_loading_nec5.py"

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)

# Ladders, in TOTAL segments. Every rung is a multiple of 4 so the mid-element
# load site (N/4) and the feed (N/2) are knots at the SAME arc length at every
# rung — the load must not walk down the wire as the mesh refines.
LADDER = (24, 32, 48, 64, 96)

DIP_LEN = 10.18946
DIP_RAD = 1.0262e-3

# The lumped load: a loading-coil-shaped R + jX, big enough to move the
# printed impedance by tens of ohms and so to be a real test of the term
# rather than of the rounding.
LOAD_R = 25.0
LOAD_X = 300.0
SIGMA_CU = 5.8e7


def dipole_wires():
    return [np.array([[0.0, 0.0, 0.0], [0.0, DIP_LEN, 0.0]])]


def _deck(n, ld_cards):
    return (
        "CM razor-loading\nCE\n"
        f"GW 1 {n} 0.000000E+00 0.000000E+00 0.000000E+00 "
        f"0.000000E+00 {DIP_LEN:.6E} 0.000000E+00 {DIP_RAD:.6E}\n"
        "GE 0\n"
        + ld_cards
        + f"EX 0 1 {n // 2} 2 1.000000E+00 0.000000E+00\n"
        + f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00\nXQ 0\nEN\n"
    )


def _ld4(seg):
    return f"LD 4 1 {seg} {seg} {LOAD_R:.6E} {LOAD_X:.6E}\n"


# Each case: (deck cards for one N, razor kwargs for one N).
CASES = {
    "unloaded": (lambda n: "", lambda n: {}),
    "lumped-at-feed": (
        lambda n: _ld4(n // 2),
        lambda n: {"lumped_loads": [(0, DIP_LEN / 2.0, LOAD_R + 1j * LOAD_X)]},
    ),
    "lumped-mid": (
        lambda n: _ld4(n // 4),
        lambda n: {"lumped_loads": [(0, DIP_LEN / 4.0, LOAD_R + 1j * LOAD_X)]},
    ),
    "copper": (
        lambda n: f"LD 5 1 1 {n} {SIGMA_CU:.6E}\n",
        lambda n: {"wire_conductivity": SIGMA_CU},
    ),
}


def razor_z(n, case, **mode):
    from momwire import RazorSolver

    z, _ = RazorSolver(
        wires=dipole_wires(),
        n_per_edge_per_wire=[[n]],
        wire_radius=DIP_RAD,
        wavelength=WL,
        **CASES[case][1](n),
        **mode,
    ).compute_impedance()
    return complex(z)


# --------------------------------------------------------------------------
def main() -> None:
    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("RAZOR_LOADING_CAPTURES", "/tmp/razor-loading-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    rows: dict[str, list[tuple[int, complex, complex, complex]]] = {}
    for name in CASES:
        out = []
        for n in LADDER:
            zn = complex(eng.run_deck(_deck(n, CASES[name][0](n)))[0][0][2])
            zq = razor_z(n, name, nec5_quadrature=True)
            zg = razor_z(n, name)
            out.append((n, zn, zq, zg))
            print(f"{name:>15} N={n:<3} nec5={zn:.4f} n5q={zq:.4f} gl={zg:.4f}")
        rows[name] = out

    _write_golden(rows)
    _report(rows)


def _write_golden(rows) -> None:
    lines = [
        '"""NEC-5 printed impedances for a loaded dipole, and the razor lanes.',
        "",
        "GENERATED by scripts/capture_razor_loading_nec5_lane.py — do not edit",
        "by hand. See that script for the decks, the LD-convention",
        "verification and the bar these numbers are gated against",
        "(momwire#427). Citation: NEC-5 (LLNL-CODE-746721), free-space",
        "ByDipole1 loaded ladders, 2026-08-17.",
        "",
        "Each row is (N_total, Z_nec5, Z_razor_n5q, Z_razor_gl) for one case:",
        "`unloaded`, `lumped-at-feed` (LD 4 on the fed segment),",
        "`lumped-mid` (the same LD 4 at the N/4 knot) and `copper` (LD 5",
        "conductivity over the whole wire). Only the n5q lane is gated, and",
        "what is gated is the DIFFERENCE from the unloaded column.",
        '"""',
        "",
        f"LOAD_Z = {LOAD_R} + {LOAD_X}j",
        # `repr`, not an exponent format: `ruff format` rewrites `5.8e+07`
        # and a re-capture would then show up as a formatting diff.
        f"SIGMA_CU = {SIGMA_CU!r}",
        "",
        "LOADED_LADDERS = {",
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
    base = {n: (zn, zq, zg) for n, zn, zq, zg in rows["unloaded"]}
    for name, out in rows.items():
        print(f"\n===== {name} =====")
        print(
            f"{'N':>4} | {'NEC-5 R':>9} {'X':>9} | {'n5q R':>9} {'X':>9} | "
            f"{'abs dR':>8} {'abs dX':>8} | {'incr dR':>8} {'incr dX':>8} | "
            f"{'|incr|':>7}"
        )
        res = []
        for n, zn, zq, zg in out:
            d = zq - zn
            bn, bq, _bg = base[n]
            incr = (zq - bq) - (zn - bn)
            res.append(incr)
            print(
                f"{n:>4} | {zn.real:9.4f} {zn.imag:9.4f} | "
                f"{zq.real:9.4f} {zq.imag:9.4f} | "
                f"{d.real:8.4f} {d.imag:8.4f} | "
                f"{incr.real:8.4f} {incr.imag:8.4f} | {abs(incr):7.4f}"
            )
        if name != "unloaded":
            print(f"  worst |increment gap|: {max(abs(r) for r in res):.4f} ohm")


if __name__ == "__main__":
    main()
