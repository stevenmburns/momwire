"""Capture the razor EXTENDED-KERNEL fat-wire lane against the NEC-5 binary.

momwire#436. The taper study of 2026-08-18 identified the NEC-5 binary as
extended-kernel EVERYWHERE and so overturned `RazorSolver`'s EK refusal; this
script is the standing measurement of what that bought. It runs three of the
study's four decks through the binary and through `RazorSolver` in four
combinations (kernel x quadrature lane) plus `BSplineSolver(degree=1)` with
the same kernel, then regenerates `tests/golden_razor_taper_nec5.py` as pure
literals so the momwire test suite needs neither the binary nor antennaknobs.

Must run under the antennaknobs venv (that is where the NEC-5 engine wrapper
lives), with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_razor_taper_nec5_lane.py

Only the binary's PRINTED impedances are recorded; nothing about NEC-5's
internals is read, quoted or inferred here.

The decks (14.2 MHz, free space, ten colinear `GW` sections each):

  fat    uniform 25 mm radius — the fattest section of Ward Harriman AE6TY's
         20:1 tapered dipole, taken uniform. a/lambda = 1.2e-3. This is the
         CONTROL: no radius step, so it measures the kernel and nothing else,
         and it is the deck on which the reduced-kernel twin claim failed the
         offset-constancy bar by 43x.
  ward   Ward's deck itself: the same ten sections with the radius stepping
         linearly 1.25 -> 25 mm. Nine radius steps, across which momwire's
         eligibility rule declines to extend and NEC still extends some
         cross-arm pairs, which is why its constancy is looser than `fat`'s.
  thin   uniform 1.25 mm — Ward's thinnest section. a/lambda = 5.9e-5, where
         the two kernels agree and the reduced row remains the twin.

THE FEED. NEC-2's `EX` drives a segment CENTRE; NEC-5's `EX` field 4 is an
END CODE and drives a KNOT. On a shared mesh those are half an element
apart, and charging momwire for the difference produced a 2.6 ohm phantom
defect in the study before it was found. Every rung here uses an EVEN
per-section count so the fed section's midpoint IS a knot, drives that knot
on the NEC-5 side (`EX 0 tag n/2 2`), and builds momwire directly with
`feed_arclength` at the same point. What is measured is formulation against
formulation.

THE LADDER STOPS AT N = 200 for the gated rungs. At N = 400 the fat end sits
at Delta/a = 1.05, below the Delta/a ~ 2 floor momwire#248 established and
below where the extended kernel's own moment expansion stops improving
(momwire#249). Rungs past that are the study's business, not a gate's.

The bar, in the shape the twin claim can honestly hold (momwire#316's
`nec5_quadrature` sharp lane):

  the (razor_EK_n5q - NEC-5) offset CONSTANT down each ladder to within
  0.05 ohm in dR and dX, on `fat`; on `ward` the same bar in dR, with dX
  recorded (the eligibility rule's documented conservatism at a radius step
  costs it there).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden_razor_taper_nec5.py"

FREQ_MHZ = 14.2
WL = 299792458.0 / (FREQ_MHZ * 1e6)

# Ward's deck: ten colinear sections over 10.51010 m, radius stepping
# linearly 1.25 mm -> 25 mm, fed at the centre of section 6.
WARD_LEN = 10.51010
WARD_NSEC = 10
WARD_SEC = WARD_LEN / WARD_NSEC
WARD_RADII = (
    1.250000e-03,
    3.888889e-03,
    6.527778e-03,
    9.166667e-03,
    1.180556e-02,
    1.444444e-02,
    1.708333e-02,
    1.972222e-02,
    2.236111e-02,
    2.500000e-02,
)
FED_SEC = 6  # 1-based tag
Z_HEIGHT = 10.0  # free space; the y/z offset is the study's, and immaterial

# Segments PER SECTION. Even, so the fed section's midpoint is a knot on
# both sides of the comparison. N_total is ten times this.
LADDER = (2, 4, 8, 14, 20)


def sections(name):
    if name == "ward":
        radii = WARD_RADII
    elif name == "fat":
        radii = (2.500000e-02,) * WARD_NSEC
    elif name == "thin":
        radii = (1.250000e-03,) * WARD_NSEC
    else:  # pragma: no cover - script
        raise ValueError(name)
    return tuple((k * WARD_SEC, (k + 1) * WARD_SEC, radii[k]) for k in range(WARD_NSEC))


GEOMS = ("fat", "ward", "thin")


def wires_of(name):
    secs = sections(name)
    wires = [
        np.array([[x0, 0.0, Z_HEIGHT], [x1, 0.0, Z_HEIGHT]]) for x0, x1, _r in secs
    ]
    radii = [r for _a, _b, r in secs]
    return wires, radii


def nec5_deck(name, n_per_sec):
    """NEC-5's spelling. No `EK` card — the binary rejects one, its
    formulation not offering the choice — and `EX` field 4 is the END CODE
    naming which knot of the segment hosts the source. With an even count the
    fed section's centre knot is end 2 of segment n/2."""
    lines = [f"CM razor taper {name} n={n_per_sec}", "CE"]
    for k, (x0, x1, rad) in enumerate(sections(name)):
        lines.append(
            f"GW {k + 1} {n_per_sec} {x0:.6E} 0.000000E+00 {Z_HEIGHT:.6E} "
            f"{x1:.6E} 0.000000E+00 {Z_HEIGHT:.6E} {rad:.6E}"
        )
    lines.append("GE 0")
    lines.append(f"EX 0 {FED_SEC} {n_per_sec // 2} 2 1.000000E+00 0.000000E+00")
    lines.append(f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00")
    lines.append("XQ 0")
    lines.append("EN")
    return "\n".join(lines) + "\n"


def momwire_z(name, n_per_sec, kind):
    from momwire import BSplineSolver, RazorSolver

    wires, radii = wires_of(name)
    fed = FED_SEC - 1
    common = dict(
        wires=wires,
        n_per_edge_per_wire=[[n_per_sec]] * len(wires),
        wire_radius=radii,
        wavelength=WL,
        feed_wire_index=fed,
        feed_arclength=WARD_SEC / 2.0,
    )
    if kind == "ek_n5q":
        s = RazorSolver(extended_kernel=True, nec5_quadrature=True, **common)
    elif kind == "red_n5q":
        s = RazorSolver(nec5_quadrature=True, **common)
    elif kind == "ek_gl":
        s = RazorSolver(extended_kernel=True, **common)
    elif kind == "red_gl":
        s = RazorSolver(**common)
    elif kind == "bs1_ek":
        junctions = [[(k, "end"), (k + 1, "start")] for k in range(len(wires) - 1)]
        s = BSplineSolver(degree=1, extended_kernel=True, junctions=junctions, **common)
    elif kind == "bs1_red":
        junctions = [[(k, "end"), (k + 1, "start")] for k in range(len(wires) - 1)]
        s = BSplineSolver(degree=1, junctions=junctions, **common)
    else:  # pragma: no cover - script
        raise ValueError(kind)
    z, _ = s.compute_impedance()
    return complex(z)


COLUMNS = ("ek_n5q", "red_n5q", "ek_gl", "red_gl", "bs1_ek", "bs1_red")


def main() -> None:
    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(os.environ.get("RAZOR_TAPER_CAPTURES", "/tmp/razor-taper-captures"))
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    rows = {}
    for name in GEOMS:
        out = []
        for n in LADDER:
            zn = complex(eng.run_deck(nec5_deck(name, n))[0][0][2])
            cols = tuple(momwire_z(name, n, c) for c in COLUMNS)
            out.append((n * WARD_NSEC, zn) + cols)
            print(
                f"{name:>5} N={n * WARD_NSEC:<4} nec5={zn:.4f} "
                f"ek_n5q={cols[0]:.4f} d={abs(cols[0] - zn):.4f}",
                flush=True,
            )
        rows[name] = out

    _write_golden(rows)
    _report(rows)


def _write_golden(rows) -> None:
    lines = [
        '"""NEC-5 printed impedances on the taper decks, and the razor kernels.',
        "",
        "GENERATED by scripts/capture_razor_taper_nec5_lane.py — do not edit",
        "by hand. See that script for the decks, the matched-feed recipe and",
        "the bar these numbers are gated against (momwire#436). Citation:",
        "NEC-5 (LLNL-CODE-746721), taper ladder decks, 2026-08-18.",
        "",
        "Each row is",
        "",
        "    (N_total, Z_nec5, ek_n5q, red_n5q, ek_gl, red_gl, bs1_ek, bs1_red)",
        "",
        "with the momwire columns `RazorSolver` under the four (kernel x",
        "quadrature lane) combinations and `BSplineSolver(degree=1)` under",
        "both kernels — the cross-formulation control, the two rows sharing",
        "one kernel. `ek_n5q` is the gated fat-wire twin lane.",
        '"""',
        "",
        "TAPER_LADDERS = {",
    ]

    def _lit(z, places):
        # Spaced operators and one column per line, so the generated file is
        # already `ruff format` clean and a re-capture never shows up as a
        # formatting diff (the row is far too wide to fit on one line).
        sign = "-" if z.imag < 0 else "+"
        return f"{z.real:.{places}f} {sign} {abs(z.imag):.{places}f}j"

    for name, out in rows.items():
        lines.append(f'    "{name}": (')
        for row in out:
            lines.append("        (")
            lines.append(f"            {row[0]},")
            lines.append(f"            {_lit(row[1], 4)},")
            for z in row[2:]:
                lines.append(f"            {_lit(z, 6)},")
            lines.append("        ),")
        lines.append("    ),")
    lines.append("}")
    GOLDEN.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {GOLDEN}")


def _report(rows) -> None:
    for name, out in rows.items():
        print(f"\n===== {name} =====")
        print(
            f"{'N':>5} | {'NEC-5':>19} | {'dZ ek_n5q':>19} | "
            f"{'dZ red_n5q':>19} | {'|ek-red|':>9} | {'|ek-bs1ek|':>10}"
        )
        de, dr = [], []
        for n, zn, ek_n5q, red_n5q, ek_gl, red_gl, bs1_ek, bs1_red in out:
            de.append(ek_n5q - zn)
            dr.append(red_n5q - zn)
            print(
                f"{n:>5} | {zn.real:9.4f}{zn.imag:+9.4f}j | "
                f"{de[-1].real:9.4f}{de[-1].imag:+9.4f}j | "
                f"{dr[-1].real:9.4f}{dr[-1].imag:+9.4f}j | "
                f"{abs(ek_n5q - red_n5q):9.4f} | {abs(ek_gl - bs1_ek):10.4f}"
            )
        for label, ds in (("EK n5q", de), ("reduced n5q", dr)):
            sr = max(x.real for x in ds) - min(x.real for x in ds)
            sx = max(x.imag for x in ds) - min(x.imag for x in ds)
            print(f"  {label:>12} offset constancy: dR {sr:.4f}, dX {sx:.4f}")


if __name__ == "__main__":
    main()
