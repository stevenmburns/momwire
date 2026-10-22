"""Point-matched two-radius candidates on the U5 rod (crossing_deck(2)),
harness-side patches only. Candidates:
  W  crossing edge weight: A shipped a_i*k_j/k_i | B a_j*k_j/k_i
  P  point-row radius: obs (observer's, shipped) | below (both at a_below)
     | src (source axis's radius) | above (both at a_above)
Prints the U5 guard metrics per candidate. Run from repo root."""

from __future__ import annotations

import itertools
import sys
import warnings

import numpy as np

sys.path.insert(0, "tests")
from test_crossing_serve_524 import crossing_deck

import momwire.sinusoidal as S
from momwire import _crossing_fill

A = 0.25e-3
warnings.simplefilter("ignore")
_W = S._edge_weight
_POB = _crossing_fill.point_observer_block


def w_b(a_const, k, basis, seg):
    a_j = a_const[seg]
    k_i, k_j = k[basis], k[seg]
    return np.where(k_i != k_j, a_j * k_j / k_i, a_j)


def w_c(a_const, k, basis, seg):
    """Radius-only NEC charge ratio on top of the shipped two-k weight."""
    g = 0.5772156649015329
    k_i, k_j = k[basis], k[seg]
    a_rad = 2.0 / (k * np.exp(1.0 / a_const + g))  # each segment's radius
    a_i_at_kj = 1.0 / (np.log(2.0 / (k_j * a_rad[basis])) - g)
    ratio = a_const[seg] / a_i_at_kj
    return np.where(k_i != k_j, a_const[basis] * k_j / k_i * ratio, a_const[seg])


def pob_for(mode):
    def pob(ctx, *a, observers_above, **kw):
        if ctx.a_below is not None or mode == "obs":
            pass
        if ctx.a_above is not None and mode != "obs":
            r = {
                "below": ctx.a_below,
                "above": ctx.a_above,
                "src": ctx.a_below if observers_above else ctx.a_above,
            }[mode]
            ctx = ctx._replace(a_wire=float(r))
        return _POB(ctx, *a, observers_above=observers_above, **kw)

    return pob


def z(ab, aa):
    s = S.SinusoidalSolver(**crossing_deck(2, wire_radius=[ab, aa]))
    out = s.compute_impedance()
    return complex(out[0] if isinstance(out, tuple) else out)


for wname, pmode in itertools.product(("A", "C"), ("obs", "src", "above")):
    S._edge_weight = {"A": _W, "B": w_b, "C": w_c}[wname]
    _crossing_fill.point_observer_block = pob_for(pmode)
    try:
        e = z(A, A)
        r2, r4, t2 = z(A / 2, A), z(A / 4, A), z(A, A / 2)
        print(
            f"W={wname} P={pmode:5s} equal {e:.3f} rise2 {r2.real - e.real:+.3f} "
            f"rise4 {r4.real - e.real:+.3f} top2 {t2.real - e.real:+.3f} "
            f"X: rise2 {r2.imag - e.imag:+.3f} top2 {t2.imag - e.imag:+.3f}",
            flush=True,
        )
    finally:
        S._edge_weight = _W
        _crossing_fill.point_observer_block = _POB
