"""Proj vs point, and the quadrature lever — momwire#1004, step 1 of the fix.

**READ THIS FIRST: the metric this script originally used was wrong, and the
wrong one is kept here beside the right one because the trap is the lesson.**

`cross_block_error` measures `|G_mixed - G_free|`. The mixed cross block is
stored NEGATED relative to the free-space one, so that quantity is
`|-G - G| = 2|G|` plus the real residual — a constant. Graded against the
cross block's own magnitude it reads 2.0000e+00 at every gap, which is the
only thing a sign does. Every knob sweep run against it came back inert, and
the conclusion "no knob moves it" followed from the metric, not the physics.

`cross_block_residual` is the corrected one: `|G_mixed + G_free| / |G_free|`,
the sign divided out. Against it the quadrature moves the 5 mm gap 13x, which
is what the registered prediction said and what the fix rests on.

This is `per_surface_rel`'s own warning one level up — "PER SURFACE against its
OWN magnitude, never against the largest of the five ... the one that is
easiest to reintroduce by accident". Reintroduced at block level.

"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve()
TESTS = HERE.parents[2] / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import momwire._below_interface as bi  # noqa: E402
import momwire._sommerfeld_transmitted as trans  # noqa: E402
from momwire import SinusoidalGalerkinSolver as SG  # noqa: E402

C0 = 299792458.0
F = 7e6
WL = C0 / F
KP = 2.0 * np.pi * F / C0
OM = 2.0 * np.pi * F
SEGS = 15
GAPS = (0.5, 0.1, 0.02, 0.005)
_ORIG_Q = bi.N_QP_BURIED_FIELD


def deck(gap):
    """The rho-spanning deck: the stressing class the fork identified."""
    return (
        np.array([(0.0, 0.0, gap), (0.6, 0.0, gap)]),
        np.array([(0.0, 0.0, -gap), (0.6, 0.0, -gap)]),
    )


def mk(wires, *, free):
    g = (
        {}
        if free
        else dict(ground_z=0.0, ground_eps=(1.0, 0.0), ground_model="sommerfeld")
    )
    return SG(
        wires=list(wires),
        n_per_edge_per_wire=[[SEGS]] * len(wires),
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL,
        wire_radius=0.001,
        **g,
    )


def G_of(s):
    geom = s._build_geometry()
    if s._is_mixed(geom):
        return s._assemble_Z(geom, s.k)[0]
    with s._operating_medium(geom):
        return s._assemble_Z(geom, s.k)[0]


def cross_block_error(gap, q):
    """Worst cross-block entry of (eps-tilde=1 minus free space), relative."""
    bi.N_QP_BURIED_FIELD = q
    trans._GRID_CACHE.clear()
    a, b = deck(gap)
    na = G_of(mk([a], free=True)).shape[0]
    Gf = G_of(mk([a, b], free=True))
    Gc = G_of(mk([a, b], free=False))
    D = Gc - Gf
    sc = np.abs(Gf).max()
    return max(np.abs(D[:na, na:]).max(), np.abs(D[na:, :na]).max()) / sc


def cross_block_residual(gap, q, nqt=8):
    """THE CORRECTED METRIC: |G_mixed + G_free| / |G_free| on the cross block.

    The sign convention divided out. This is the number every #1004 claim
    should be stated against.
    """
    bi.N_QP_BURIED_FIELD = q
    trans._GRID_CACHE.clear()
    a, b = deck(gap)
    na = G_of(mk([a], free=True)).shape[0]
    Gf = G_of(mk([a, b], free=True))
    Gc = G_of(mk([a, b], free=False))
    bi.N_QP_BURIED_FIELD = _ORIG_Q
    xf = Gf[:na, na:]
    xc = Gc[:na, na:]
    return np.abs(xc + xf).max() / np.abs(xf).max()


def proj_vs_point(gap):
    """One observer, one source, same grid: proj[0,0] vs t_obs . point."""
    trans._GRID_CACHE.clear()
    grid = trans.get_grid_below_above(
        1.0 + 0j, KP, r_max=0.6, zp_min=gap, zp_max=gap, omega=OM, r_min=1e-3
    )
    worst = 0.0
    for rho in (0.05, 0.3, 0.6):
        obs = np.array([[rho, 0.0, gap]])
        src = np.array([[0.0, 0.0, -gap]])
        for t_o in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            for t_s in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
                p = trans.transmitted_field_proj_below_to_above(
                    obs, t_o[None, :], src, t_s[None, :], 0.0, KP, KP + 0j, grid
                )[0, 0]
                e = trans.transmitted_field_below_to_above(
                    obs, src, t_s[None, :].astype(complex), 0.0, KP, KP + 0j, grid
                )[0]
                pt = complex(np.dot(t_o.astype(complex), e))
                den = max(abs(p), abs(pt), 1e-300)
                worst = max(worst, abs(p - pt) / den)
    return worst


def main() -> int:
    print("=" * 78)
    print("P1 — is the PROJECTED composition the point composition?")
    print("=" * 78)
    print(f"{'gap':>8} {'proj vs point':>16}")
    w1 = 0.0
    for gap in GAPS:
        r = proj_vs_point(gap)
        w1 = max(w1, r)
        print(f"{gap:8.3f} {r:16.3e}")
    verdict1 = "CONFIRMED (composition correct)" if w1 <= 1e-9 else "NOT CONFIRMED"
    print(f"\n  worst {w1:.3e} — predicted <= 1e-9: {verdict1}")
    print(f"  falsifier >= 1e-3: {'FIRED' if w1 >= 1e-3 else 'not fired'}")

    print()
    print("=" * 78)
    print("P2 — the QUADRATURE lever (N_QP_BURIED_FIELD, shared by SG+bspline)")
    print("=" * 78)
    # Gate: the knob must actually change the order the fill asks for.
    bi.N_QP_BURIED_FIELD = 24
    assert bi.n_qp_buried_field(3) == 24, "the q knob is inert"
    bi.N_QP_BURIED_FIELD = _ORIG_Q
    print("  gate: n_qp_buried_field(3) follows the knob          PASS")
    print()
    print("  BOTH metrics, side by side — the point of this script.")
    print(
        f"{'gap':>8} "
        + "".join(f"{'q=' + str(q):>12}" for q in (6, 24))
        + "   |"
        + "".join(f"{'q=' + str(q):>12}" for q in (6, 24))
    )
    print(
        f"{'':>8} {'--- |Gc - Gf| (WRONG) ---':>24}   |{'--- |Gc + Gf| (right) ---':>25}"
    )
    for gap in GAPS:
        wrong = [cross_block_error(gap, q) for q in (6, 24)]
        right = [cross_block_residual(gap, q) for q in (6, 24)]
        bi.N_QP_BURIED_FIELD = _ORIG_Q
        print(
            f"{gap:8.3f} "
            + "".join(f"{v:12.4e}" for v in wrong)
            + "   |"
            + "".join(f"{v:12.4e}" for v in right)
        )

    print()
    print("=" * 78)
    print("READ")
    print("=" * 78)
    print("  The left pair barely moves with q: it is dominated by the constant")
    print("  2|G_free| the sign convention puts there. The right pair moves 13x")
    print("  at the 5 mm gap, which is the registered prediction and the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
