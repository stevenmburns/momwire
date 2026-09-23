"""momwire#1154 probe 1: which half of the coated-wire PAIR is wrong in soil.

Reference-free, real constructor (BSplineSolver), a horizontal buried dipole.
Real (lossless) soil eps~ so the in-medium Popovic-Nesic pair has a REAL
equivalent radius and can be served as an ordinary bare wire:

  exact   bare wire of radius a'(eps~) = a (b/a)^(1 - eps~/eps_r)
          + series L = mu0/2pi (1 - eps~/eps_r) ln(b/a)     (DistributedRLC)
  today   the jacket kwargs: a'_fs = a (b/a)^(1 - 1/eps_r), L_fs
  issue   a'_fs kernel + L(eps~)  (the issue's proposed L-only change)
  charge  today + a local ELASTANCE term (the derivation in NOTES.md):
          Z += dS'/(j w) * int f_m' f_n' dl,
          dS' = ln(b/a) / (2 pi eps0 eps_r) * (1 - 1/eps~)

eps_r(jacket) = 10 > eps~ = 4 keeps L(eps~) >= 0 so DistributedRLC takes it.
"""

import numpy as np

from momwire import _ground_refl
from momwire._wire_loading import MU0, DistributedRLC
from momwire.bspline import BSplineSolver

EPS0 = 8.854187817e-12
WL = 299792458.0 / 7.0e6
A, B, ER = 1.0e-3, 3.0e-3, 10.0


def deriv_gram(s):
    """COO of int f_m' f_n' dl per segment, tagged by segment."""
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    n_b, n_w, n_p = polys.shape
    dp = polys[:, :, 1:] * np.arange(1, n_p)[None, None, :]
    h = geom["h_per_seg"]
    seg_map = {}
    nz = np.any(polys != 0.0, axis=2)
    for m in range(n_b):
        for a in range(n_w):
            if nz[m, a]:
                seg_map.setdefault(int(supp_seg[m, a]), []).append((m, a))
    pq = np.arange(n_p - 1)
    ps = pq[:, None] + pq[None, :] + 1
    rows, cols, vals, segs = [], [], [], []
    for sg, ent in seg_map.items():
        H = h[sg] ** ps / ps
        C = dp[[m for m, _ in ent], [a for _, a in ent], :]
        M = C @ H @ C.T
        for i, (mi, _) in enumerate(ent):
            for j, (mj, _) in enumerate(ent):
                rows.append(mi)
                cols.append(mj)
                vals.append(M[i, j])
                segs.append(sg)
    return np.array(rows), np.array(cols), np.array(vals), np.array(segs)


def install_charge(monkeypatch_target):
    orig = BSplineSolver._apply_loading

    def patched(self, Z, omega=None):
        Z = orig(self, Z, omega)
        if self.insulation_radius is None or Z.ndim != 2:
            return Z
        om = self.omega if omega is None else omega
        et = _ground_refl.eps_tilde(self.ground_eps, om, self.eps)
        a = self._conductor_radius_per_wire[0]
        b = self.insulation_radius[0]
        er = self.insulation_eps_r[0]
        dS = np.log(b / a) / (2 * np.pi * EPS0 * er) * (1 - 1 / et)
        r, c, v, _ = deriv_gram(self)
        np.add.at(Z, (r, c), dS / (1j * om) * v)
        return Z

    monkeypatch_target.setattr(BSplineSolver, "_apply_loading", patched)


def dipole(n, length, depth, et, **kw):
    pts = np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    arc = 0.5 * length  # knot for even n
    return BSplineSolver(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, arc, 1 + 0j)],
        wavelength=WL,
        ground_z=0.0,
        ground_eps=et,
        ground_model="sommerfeld",
        **kw,
    )


def zin(**kw):
    return complex(dipole(**kw).compute_impedance()[0])


class MP:
    def __init__(self):
        self.saved = []

    def setattr(self, obj, name, val):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for obj, name, val in reversed(self.saved):
            setattr(obj, name, val)
        self.saved = []


def run(et_val, n, length, depth):
    et = complex(et_val, -1e-9)
    lnba = np.log(B / A)
    a_ex = A * (B / A) ** (1 - et_val / ER)
    L_ex = MU0 / (2 * np.pi) * (1 - et_val / ER) * lnba
    a_fs = A * (B / A) ** (1 - 1 / ER)
    base = dict(n=n, length=length, depth=depth, et=et)
    z_exact = zin(
        wire_radius=a_ex, distributed_rlc=DistributedRLC("series", l=L_ex), **base
    )
    z_today = zin(wire_radius=A, insulation_radius=B, insulation_eps_r=ER, **base)
    z_issue = zin(
        wire_radius=a_fs, distributed_rlc=DistributedRLC("series", l=L_ex), **base
    )
    mp = MP()
    install_charge(mp)
    try:
        z_charge = zin(wire_radius=A, insulation_radius=B, insulation_eps_r=ER, **base)
    finally:
        mp.undo()
    z_bare = zin(wire_radius=A, **base)
    return z_exact, z_today, z_issue, z_charge, z_bare


if __name__ == "__main__":
    for et_val in (4.0, 8.0):
        for length, n in ((1.0, 20), (1.0, 40), (1.0, 80), (5.0, 40), (5.0, 80)):
            ze, zt, zi, zc, zb = run(et_val, n, length, 0.5)
            print(
                f"eps~={et_val:4.1f} L={length:3.1f} n={n:3d}  exact {ze:.4f}\n"
                f"   today-exact {zt - ze:.4f}  issue-exact {zi - ze:.4f}  "
                f"charge-exact {zc - ze:.4f}   (bare {zb:.4f})"
            )
