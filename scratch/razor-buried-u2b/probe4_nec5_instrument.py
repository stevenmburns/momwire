"""Probe 4 (momwire#1149 U2b): the INSTRUMENT, not a gate -- razor-2p on the
two-radius rod against the licensed NEC-5 reference at EQUAL mesh, every
edge refined, soil A, feed on the knot at above arclength 4.5. The reference
is a black box, verified against our licensed materials; nothing about its
internals is used or cited.

Needs antennaknobs on PYTHONPATH for its deck runner (read-only):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<ak>/src .venv/bin/python probe4...
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from antennaknobs.engines.nec5 import NEC5Engine, run_deck
from test_crossing_serve_524 import crossing_deck

import momwire
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
EXE = str(Path.home() / "antennas/NEC5-downloads/nec5-linux/nec5cl")
A, EPS, SIG, F_MHZ = 0.25e-3, 13.0, 0.005, 7.0
ZS = [
    (-2.0, -0.5),
    (-0.5, -0.1),
    (-0.1, -0.025),
    (-0.025, 0.0),
    (0.0, 0.025),
    (0.025, 0.1),
    (0.1, 0.5),
    (0.5, 10.0),
]


def n(x):
    return f"{float(x):.6E}"


def nec5_deck(counts, a_below, a_above):
    lines = ["CM probe4 two-radius rod", "CE"]
    for tag, ((z0, z1), ns) in enumerate(zip(ZS, counts, strict=True), start=1):
        a = a_below if z1 <= 0.0 else a_above
        lines.append(
            f"GW {tag} {ns} {n(0)} {n(0)} {n(z0)} {n(0)} {n(0)} {n(z1)} {n(a)}"
        )
    lines.append("GE -1 0")
    lines.append(f"GN 0 0 0 0 {n(EPS)} {n(SIG)} {n(1.0)} {n(0.0)} NOFILE")
    seg = round(counts[7] * 4.0 / 9.5)
    lines.append(f"EX 0 8 {seg} 2 {n(1.0)} {n(0.0)}")
    lines.append(f"FR 0 1 0 0 {n(F_MHZ)} {n(0.0)}")
    lines.append("XQ 0")
    lines.append("EN")
    return "\n".join(lines) + "\n"


CASES = (
    ("equal", A, A),
    ("rise/2", A, A / 2),
    ("rise/4", A, A / 4),
    ("top/4", A / 4, A),
)
for m in (1, 2, 4, 8):
    res = {}
    for name, a_above, a_below in CASES:
        d = crossing_deck(2, wire_radius=[a_below, a_above])
        b, a = d["n_per_edge_per_wire"]
        b, a = [x * m for x in b], [x * m for x in a]
        d["n_per_edge_per_wire"] = [b, a]
        d["feeds"] = [(1, 4.5, 1 + 0j)]
        d.pop("junctions")
        zr = complex(RazorSolver(**d, nec5_quadrature=True).compute_impedance()[0])
        txt = run_deck(EXE, nec5_deck(b + a, a_below, a_above), timeout=600)
        zn = NEC5Engine._parse_input_parameters(txt, current=False)[0][0][2]
        res[name] = (zr, zn)
        dz = zr - zn
        print(
            f"x{m:<2d} {name:7s} razor {zr.real:9.4f}{zr.imag:+9.4f}j  "
            f"ref {zn.real:9.4f}{zn.imag:+9.4f}j  |d| {abs(dz):.4f}",
            flush=True,
        )
    for name in ("rise/2", "rise/4", "top/4"):
        dr = res[name][0].real - res["equal"][0].real
        dn = res[name][1].real - res["equal"][1].real
        print(
            f"x{m:<2d} {name:7s} dR razor {dr:+8.4f}  ref {dn:+8.4f}  diff {dr - dn:+.4f}",
            flush=True,
        )
