"""momwire#1156 probe 1: is every SG basis continuous across the interior
segment boundaries of its support, at real AND complex k?

The Galerkin charge term zq * int f_i' f_j' dl (bspline's `_charge_gram`
weak form) is only the whole term if f has no jump inside a wire: a jump
would put a delta in f' that a per-segment integral cannot see. Also
reports each basis's value at the wire's free ends (the by-parts boundary
term there is f * (dS' q), dropped by the weak form)."""

import numpy as np

from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WL = 299792458.0 / 7.0e6
pts = np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)])
s = SinusoidalGalerkinSolver(
    wires=[pts, pts + [0, 1.0, 0]],
    n_per_edge_per_wire=[[11], [7]],
    feeds=[(0, 2.5, 1 + 0j)],
    wavelength=WL,
    wire_radius=1e-3,
)
geom = s._build_geometry()
k0 = 2 * np.pi / WL
for k in (k0, k0 * complex(3.6, -0.9)):
    v = s._basis_coefs(geom, k)
    starts = np.asarray(v["starts"])
    h = np.asarray(geom["seg_h"])
    N = int(geom["n_segs"])
    val = {}  # (basis, seg) -> (f(-h/2), f(+h/2))
    for m in range(N):
        for e in range(starts[m], starts[m + 1]):
            P = v["sigma"][e] * v["A"][e]
            Q = v["B"][e]
            R = v["sigma"][e] * v["C"][e]
            ends = [
                P + Q * np.sin(k * x) + R * np.cos(k * x) for x in (-h[m] / 2, h[m] / 2)
            ]
            val[(int(v["jbasis"][e]), m)] = tuple(ends)
    first = np.asarray(geom["wire_first"])
    last = np.asarray(geom["wire_last"])
    jump = 0.0
    scale = 0.0
    ends = 0.0
    for (j, m), (lo, hi) in val.items():
        scale = max(scale, abs(lo), abs(hi))
        w = int(np.searchsorted(first, m, side="right") - 1)
        nxt = val.get((j, m + 1))
        if m < last[w]:
            right = nxt[0] if nxt is not None else 0.0
            jump = max(jump, abs(hi - right))
        if m > first[w] and (j, m - 1) not in val:
            jump = max(jump, abs(lo))
        if m == first[w]:
            ends = max(ends, abs(lo))
        if m == last[w]:
            ends = max(ends, abs(hi))
    print(
        f"k={k:.5f}: max interior jump {jump:.3e} (scale {scale:.3e}); "
        f"max |f| at free ends {ends:.3e}"
    )
