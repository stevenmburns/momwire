"""momwire#282 stage 2, experiment §5.4-(1): does the low-eps_r contact gap
survive when the Sommerfeld remainder is evaluated DIRECTLY, bypassing the
interpolation grid entirely, at a tolerance far tighter than production's?

The study's wording: "recompute the near-diagonal remainder blocks by direct
evaluation at very high rtol, bypassing the grid entirely, at N = 41 poor
soil, and see whether the 3.3 ohm moves. If it does, this is it and it is a
quadrature/asymptotics problem in Q."

`DirectGrid` is a drop-in for `SommerfeldGrid` on the numpy remainder path:
same `eval(R1, theta)` contract, answering with `iv_surfaces_direct` at the
requested rtol instead of a cubic Lagrange stencil. Only the two
grid-consuming accelerator kernels are masked, so the numpy body that calls
`grid.eval` runs while the rest of the fill stays accelerated.

It is cheap on a vertical because every pair collapses onto rho = 0, so the
distinct (R1, theta) arguments number in the hundreds rather than the tens
of thousands: N = 81 poor soil is 1,691 direct evaluations and ~0.2 s.

WHAT IT MEASURED (2026-08-19, momwire#282 stage 2)
---------------------------------------------------
Candidate 1 is dead. With the grid gone and rtol at 1e-11, the monopole's
poor-soil residual against the binary moves 3.3274 -> 3.3305 ohm at N = 81
and 3.2691 -> 3.2722 at N = 41; average soil moves 1.2712 -> 1.2715. Raising
`n_qp_sommerfeld` 3 -> 12 at N = 41 poor moves it 3.2691 -> 3.2409. Neither
knob is worth 3 ohm, so the remainder's evaluation is not the gap.

What the grid IS worth, which is not nothing and belongs to momwire#443: on
SEA WATER the grid costs 0.13 ohm at every mesh (N = 21/41/81 residuals
0.5248/0.3721/0.2703 with the grid, 0.4445/0.2756/0.1611 without), i.e.
about 40 % of that row's decay bar. On very good ground it is 0.0037, on
average 0.0008, on poor 0.0032. That is exactly momwire#443's shape — a
near-PEC error at the small R1 only contact geometries query — showing up
in a shipped answer rather than in a limit gate.

Usage:
    python scripts/probe_contact_direct_remainder.py --n 41 --ground poor
    python scripts/probe_contact_direct_remainder.py --n 21 41 81 --ground sea
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from momwire import BSplineSolver, _sommerfeld, bspline

C = 299792458.0
FREQ_MHZ = 14.0
WL = C / (FREQ_MHZ * 1e6)
RAD = 0.005
MONO_H = 5.3535

GROUND_EPS = {
    "sea": (81.0, 5.0),
    "vgood": (20.0, 0.0303),
    "avg": (13.0, 0.005),
    "poor": (5.0, 0.001),
}

# tests/golden_contact_nec5.py, monopole rows, as (N, Z).
NEC5 = {
    "pec": {
        11: 39.8820 + 19.2600j,
        21: 40.3790 + 21.0870j,
        41: 40.6430 + 22.0180j,
        61: 40.7450 + 22.3610j,
        81: 40.8040 + 22.5510j,
    },
    "sea": {
        11: 41.8430 + 20.3150j,
        21: 42.5050 + 22.4220j,
        41: 42.8580 + 23.5730j,
        61: 42.9880 + 24.0100j,
        81: 43.0590 + 24.2500j,
    },
    "vgood": {
        11: 51.9360 + 22.2610j,
        21: 52.6620 + 24.6770j,
        41: 53.0310 + 25.9260j,
        61: 53.1700 + 26.3760j,
        81: 53.2480 + 26.6160j,
    },
    "avg": {
        11: 51.2860 + 19.2070j,
        21: 52.0060 + 21.5050j,
        41: 52.3720 + 22.6850j,
        61: 52.5080 + 23.1100j,
    },
    "poor": {
        11: 43.5710 + 15.1130j,
        21: 44.2440 + 17.1690j,
        41: 44.5870 + 18.2220j,
        61: 44.7130 + 18.6070j,
        81: 44.7830 + 18.8170j,
    },
}


class MaskedAcc:
    """The accelerator module with only its GRID-consuming kernels hidden.

    Setting `_acc = None` outright also disables `assemble_Z_bspline` and the
    rest of the fill, which the module reaches for unguarded. Both remainder
    kernels are selected by `hasattr`, so masking those two names is enough
    to route the remainder — and only the remainder — through the numpy body
    that calls `grid.eval`.
    """

    _HIDDEN = ("sommerfeld_remainder_bspline_Q", "remainder_field_proj_batch")

    def __init__(self, acc):
        self._acc = acc

    def __getattr__(self, name):
        if name in MaskedAcc._HIDDEN:
            raise AttributeError(name)
        return getattr(self._acc, name)


class DirectGrid:
    """`SommerfeldGrid`'s query contract, answered without a grid.

    Carries the same public attributes `remainder_field_proj` reads
    (`r1_max`, plus the physical constants) so it substitutes cleanly.
    Memoized on the exact (R1, theta) pair: a vertical wire's pairs collapse
    onto rho = 0 / theta = pi/2, and the symmetric obs/src halves of any
    deck repeat their arguments bit-for-bit, so the cache is a real factor
    without approximating anything.
    """

    def __init__(self, eps_t, k2, r1_max, omega, mu, rtol):
        self.eps_t = complex(eps_t)
        self.k2 = float(k2)
        self.r1_max = float(r1_max)
        self.r_near = float(r1_max)
        self.omega = float(omega)
        self.mu = float(mu)
        self.rtol = float(rtol)
        self._cache: dict[tuple[float, float], tuple[complex, ...]] = {}
        self.calls = 0
        self.misses = 0

    def eval(self, R1, theta):
        R1 = np.asarray(R1, dtype=float)
        theta = np.asarray(theta, dtype=float)
        r_b, th_b = np.broadcast_arrays(R1, theta)
        shape = r_b.shape
        r_f = r_b.ravel()
        th_f = np.clip(th_b.ravel(), 0.0, 0.5 * np.pi)
        self.calls += r_f.size

        keys = [(float(a), float(b)) for a, b in zip(r_f, th_f)]
        fresh = sorted({kk for kk in keys if kk not in self._cache})
        if fresh:
            self.misses += len(fresh)
            rr = np.array([kk[0] for kk in fresh])
            tt = np.array([kk[1] for kk in fresh])
            surf = _sommerfeld.iv_surfaces_direct(
                self.eps_t,
                self.k2,
                rr,
                tt,
                rtol=self.rtol,
                omega=self.omega,
                mu=self.mu,
            )
            for i, kk in enumerate(fresh):
                self._cache[kk] = tuple(surf[s][i] for s in _sommerfeld._SURF_KEYS)

        out = np.empty((len(_sommerfeld._SURF_KEYS), r_f.size), dtype=np.complex128)
        for i, kk in enumerate(keys):
            out[:, i] = self._cache[kk]
        return {
            key: out[s].reshape(shape) for s, key in enumerate(_sommerfeld._SURF_KEYS)
        }


def solve(n, ground, *, direct=None, n_qp=None):
    """Z of the base-fed contact monopole at mesh `n` over `ground`.

    `direct=rtol` swaps the interpolation grid for `DirectGrid` at that
    tolerance; `direct=None` is the shipped path, accelerators and all.
    """
    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_H]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=0.0,
        ground_z=0.0,
    )
    if ground != "pec":
        kw.update(ground_eps=GROUND_EPS[ground], ground_model="sommerfeld")
    if n_qp is not None:
        kw["n_qp_sommerfeld"] = n_qp

    if direct is None:
        z, _ = BSplineSolver(**kw).compute_impedance()
        return complex(z), None

    grids: list[DirectGrid] = []
    saved_acc_b, saved_acc_s = bspline._acc, _sommerfeld._acc
    saved_grid = BSplineSolver._somm_grid

    def _direct_grid(self, eps_t, r1_max):
        g = DirectGrid(eps_t, self.k, r1_max, self.omega, self.mu, direct)
        grids.append(g)
        return g

    bspline._acc = MaskedAcc(saved_acc_b) if saved_acc_b is not None else None
    _sommerfeld._acc = MaskedAcc(saved_acc_s) if saved_acc_s is not None else None
    BSplineSolver._somm_grid = _direct_grid
    try:
        z, _ = BSplineSolver(**kw).compute_impedance()
    finally:
        bspline._acc, _sommerfeld._acc = saved_acc_b, saved_acc_s
        BSplineSolver._somm_grid = saved_grid
    return complex(z), grids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[41])
    p.add_argument("--ground", nargs="+", default=["poor"])
    p.add_argument("--rtol", type=float, default=1e-11)
    p.add_argument("--n-qp", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    for n in args.n:
        pec_grid, _ = solve(n, "pec")
        for ground in args.ground:
            t0 = time.time()
            z_grid, _ = solve(n, ground, n_qp=args.n_qp)
            t_grid = time.time() - t0
            t0 = time.time()
            z_dir, grids = solve(n, ground, direct=args.rtol, n_qp=args.n_qp)
            t_dir = time.time() - t0
            d_grid = z_grid - pec_grid
            d_dir = z_dir - pec_grid
            row = dict(
                n=n,
                ground=ground,
                rtol=args.rtol,
                n_qp=args.n_qp,
                z_pec=[pec_grid.real, pec_grid.imag],
                z_grid=[z_grid.real, z_grid.imag],
                z_direct=[z_dir.real, z_dir.imag],
                shift_grid=[d_grid.real, d_grid.imag],
                shift_direct=[d_dir.real, d_dir.imag],
                grid_vs_direct=abs(z_grid - z_dir),
                unique_points=sum(g.misses for g in grids) if grids else 0,
                seconds=[t_grid, t_dir],
            )
            if n in NEC5.get(ground, {}) and n in NEC5["pec"]:
                d_n5 = NEC5[ground][n] - NEC5["pec"][n]
                row["shift_nec5"] = [d_n5.real, d_n5.imag]
                row["residual_grid"] = abs(d_grid - d_n5)
                row["residual_direct"] = abs(d_dir - d_n5)
            rows.append(row)
            print(json.dumps(row), flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
