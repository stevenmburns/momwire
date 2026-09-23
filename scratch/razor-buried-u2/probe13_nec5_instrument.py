"""Probe 13 (momwire#1149 U2): the INSTRUMENT, not a gate -- razor-2p (with
the U2 node term) against the licensed NEC-5 reference at EQUAL mesh on
crossing_deck(1), soil A, feed on the knot at z = 4.5. The reference is a
black box, verified against our licensed materials; nothing about its
internals is used or cited.

Two ladders: "node" keeps the two node-adjacent edges at 2 segments (the
scoping's probe 8 ladder), "all" refines every edge.

Needs antennaknobs on PYTHONPATH for its deck runner (read-only):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<ak>/src .venv/bin/python probe13...
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from antennaknobs.engines.nec5 import NEC5Engine, run_deck
from test_crossing_serve_524 import crossing_deck

import momwire
from momwire import razor as _razor
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True
EXE = str(Path.home() / "antennas/NEC5-downloads/nec5-linux/nec5cl")
A, EPS, SIG, F_MHZ = 0.001, 13.0, 0.005, 7.0


def n(x):
    return f"{float(x):.6E}"


def nec5_deck(counts):
    zs = [(-2.0, -0.5), (-0.5, -0.1), (-0.1, 0.0), (0.0, 0.1), (0.1, 0.5), (0.5, 10.0)]
    lines = ["CM probe13 crossing deck", "CE"]
    for tag, ((z0, z1), ns) in enumerate(zip(zs, counts, strict=True), start=1):
        lines.append(
            f"GW {tag} {ns} {n(0)} {n(0)} {n(z0)} {n(0)} {n(0)} {n(z1)} {n(A)}"
        )
    lines.append("GE -1 0")
    lines.append(f"GN 0 0 0 0 {n(EPS)} {n(SIG)} {n(1.0)} {n(0.0)} NOFILE")
    # feed at z = 4.5: the last GW (0.5 -> 10), segment counts[5] * 4/9.5 -> knot
    seg = round(counts[5] * 4.0 / 9.5)
    lines.append(f"EX 0 6 {seg} 2 {n(1.0)} {n(0.0)}")
    lines.append(f"FR 0 1 0 0 {n(F_MHZ)} {n(0.0)}")
    lines.append("XQ 0")
    lines.append("EN")
    return "\n".join(lines) + "\n"


for ladder in ("node", "all"):
    for m in (1, 2, 4, 8, 16):
        d = crossing_deck(1)
        b, a = d["n_per_edge_per_wire"]
        if ladder == "node":
            b = [x * m for x in b[:-1]] + [b[-1]]
            a = [a[0]] + [x * m for x in a[1:]]
        else:
            b = [x * m for x in b]
            a = [x * m for x in a]
        d["n_per_edge_per_wire"] = [b, a]
        d["feeds"] = [(1, 4.5, 1 + 0j)]
        d.pop("junctions")
        zr = complex(RazorSolver(**d, nec5_quadrature=True).compute_impedance()[0])
        txt = run_deck(EXE, nec5_deck(b + a), timeout=600)
        zn = NEC5Engine._parse_input_parameters(txt, current=False)[0][0][2]
        dz = zr - zn
        print(
            f"{ladder:4s} x{m:<2d} razor {zr.real:9.4f}{zr.imag:+9.4f}j  "
            f"ref {zn.real:9.4f}{zn.imag:+9.4f}j  d {dz.real:+7.4f}{dz.imag:+7.4f}j "
            f"|d| {abs(dz):.4f}",
            flush=True,
        )
