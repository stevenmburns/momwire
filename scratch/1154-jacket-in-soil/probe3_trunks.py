"""momwire#1154 probe 3: the shipped term on both trunks, real constructors.

(a) exact-pair check: lossless soil eps~ = 4, jacket eps_r = 10 (so the
    in-medium pair a'(eps~), L(eps~) is real and servable as a bare wire +
    DistributedRLC). fixed = the jacket kwargs on this branch; old = the same
    with `buried_jacket_charge` forced to None (main's free-space pair).
(b) soil A (13, 5 mS/m), PVC-class jacket: the jacket SHIFT (jacketed - bare)
    on bspline and razor under refinement, fixed and old.
"""

import sys
import warnings

import numpy as np

from momwire import _wire_loading
from momwire._wire_loading import MU0, DistributedRLC
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

warnings.simplefilter("ignore")
WL = 299792458.0 / 7.0e6


def deck(n, length, depth, eps, **kw):
    pts = np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    return dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 0.5 * length, 1 + 0j)],
        wavelength=WL,
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
        **kw,
    )


def z(cls, d, old=False):
    extra = dict(nec5_quadrature=True) if cls is RazorSolver else {}
    orig = _wire_loading.buried_jacket_charge
    if old:
        _wire_loading.buried_jacket_charge = lambda s, w: None
    try:
        return complex(cls(**d, **extra).compute_impedance()[0])
    finally:
        _wire_loading.buried_jacket_charge = orig


def part_a():
    A, B, ER, ET = 1e-3, 3e-3, 10.0, 4.0
    a_ex = A * (B / A) ** (1 - ET / ER)
    L_ex = MU0 / (2 * np.pi) * (1 - ET / ER) * np.log(B / A)
    eps = complex(ET, -1e-9)
    print("(a) exact pair, eps~=4 lossless, jacket eps_r=10, 5 m dipole 0.5 m deep")
    for cls in (BSplineSolver, RazorSolver):
        for n in (20, 40, 80):
            ze = z(
                cls,
                deck(
                    n,
                    5.0,
                    0.5,
                    eps,
                    wire_radius=a_ex,
                    distributed_rlc=DistributedRLC("series", l=L_ex),
                ),
            )
            jk = dict(wire_radius=A, insulation_radius=B, insulation_eps_r=ER)
            zf = z(cls, deck(n, 5.0, 0.5, eps, **jk))
            zo = z(cls, deck(n, 5.0, 0.5, eps, **jk), old=True)
            print(
                f"  {cls.__name__:14s} n={n:3d} exact {ze:.4f}  fixed-exact "
                f"{zf - ze:.4f}  old-exact {zo - ze:.4f}",
                flush=True,
            )


def part_b(length=5.0, ns=(20, 40, 80, 160)):
    A, B, ER = 1e-3, 1.8e-3, 3.5
    soil = (13.0, 0.005)
    jk = dict(wire_radius=A, insulation_radius=B, insulation_eps_r=ER)
    print(
        f"(b) soil A, PVC jacket b={B * 1e3} mm eps_r={ER}, {length} m dipole 0.5 m deep"
    )
    rows = {}
    for cls in (BSplineSolver, RazorSolver):
        for n in ns:
            zb = z(cls, deck(n, length, 0.5, soil, wire_radius=A))
            zf = z(cls, deck(n, length, 0.5, soil, **jk))
            zo = z(cls, deck(n, length, 0.5, soil, **jk), old=True)
            rows[(cls, n)] = (zb, zf, zo)
            print(
                f"  {cls.__name__:14s} n={n:3d} bare {zb:.3f}  fixed {zf:.3f}  "
                f"old {zo:.3f}  (fixed-old {zf - zo:.3f})",
                flush=True,
            )
    print("  bspline-vs-razor gap in the jacket shift:")
    for n in ns:
        b = rows[(BSplineSolver, n)]
        r = rows[(RazorSolver, n)]
        gf = abs((b[1] - b[0]) - (r[1] - r[0]))
        go = abs((b[2] - b[0]) - (r[2] - r[0]))
        cross = abs((b[1] - b[0]) - (r[2] - r[0]))
        print(
            f"   n={n:3d}  fixed {gf:.4f}  old {go:.4f}  "
            f"bspline-fixed vs razor-old {cross:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if "a" in which:
        part_a()
    if "b" in which:
        part_b()
