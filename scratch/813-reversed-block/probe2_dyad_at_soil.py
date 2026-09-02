"""Which zz kernel and which W sign — settled at SOIL, not at eps~ = 1.

probe1 found all four candidate spellings identical at eps~ = 1. The reason
is that W, dzW and dzpW are all exactly 0 there: the W family is a pure
interface effect and a homogeneous medium has none. So the eps~ = 1 collapse
validates U, V, the by-parts role assignment and the assembly, and is BLIND
to the dyad's mixed terms.

At soil they are all nonzero, and dzpW is not -dzW, so the spelling matters.
The discriminator that does not need a new truth: run the reversed block with
GALERKIN axes on both sides. There the two blocks are genuine transposes by
Galerkin reciprocity — that is the identity bspline's crossing fill already
rides — so the correct dyad must reproduce

    cross_complete_block(ctx, A_above, B_below).T

to roundoff. A wrong zz kernel or a wrong W sign cannot.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import RazorSolver
from momwire import _crossing_fill as CF
from test_crossing_serve_524 import crossing_deck
from test_buried_serve_553 import SOIL_A
from probe1_reversed_block import reversed_block


def main():
    deck = crossing_deck(1)
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    rs = RazorSolver(**fs, nec5_quadrature=False, n_qp_path=8)
    geom = rs._build_geometry()
    seg_off = np.asarray(geom["seg_offsets"])
    b_idx = np.arange(seg_off[0], seg_off[1])
    a_idx = np.arange(seg_off[1], seg_off[2])

    for label, ge in (("eps~ = 1", (1.0, 0.0)), ("soil A", SOIL_A)):
        ctx = rs._crossing_context(geom, ground_eps=ge)
        A = CF.axis_data(ctx, a_idx)  # above, Galerkin tents
        B = CF.axis_data(ctx, b_idx)  # below, Galerkin tents
        for corner in (False, True):
            fwd = CF.cross_complete_block(ctx, A, B, corner=corner).T
            scale = np.abs(fwd).max()
            print(
                f"\n=== {label}, corner={corner} — "
                f"Galerkin both sides, max|t_ab.T| {scale:.6e} ==="
            )
            print(
                f"  {'zz kernel':>10} {'W sign':>7} {'max|t_ba - t_ab.T|':>20} "
                f"{'relative':>12}"
            )
            for zz_key in ("dzW", "dzpW"):
                for w_sign in (+1.0, -1.0):
                    rev = reversed_block(
                        ctx, B, A, zz_key=zz_key, w_sign=w_sign, corner=corner
                    )
                    d = np.abs(rev - fwd).max()
                    print(
                        f"  {zz_key:>10} {w_sign:>+7.0f} {d:>20.6e} {d / scale:>12.4e}"
                    )


main()
