"""momwire#1029 phase 2b: the swept entry on the sector route.

Two jobs, both registered by the unit:

* `--dense` dumps `compute_impedance_swept` on decks the route does NOT
  serve, so the before/after pair proves the default swept path is
  byte-identical (gate P2B-3). One deck per swept ROUTE: a free-space
  dipole takes the fully batched fast path, the 4-radial screen with the
  flag OFF takes the per-k fallback (a finite ground disqualifies the
  batched one).
* `--route` measures the sector route's own swept answer against the
  route's per-frequency `compute_impedance` (P2B-2a) and against the DENSE
  swept path (P2B-2b), at three frequencies spanning +-10 % of the deck's
  design frequency, on the 4- and 12-radial decks.
* `--ak` drives the WHOLE consumer contract instead of the solver's:
  `antennaknobs`' `MomwireEngine.impedance_sweep` on the real
  `verticals.buried_radial_vertical` design, with and without the flag. It
  is what the AK half will call, and the only place the contract is checked
  by the code that depends on it rather than by a gate that restates it.

Run from the repo root with `tests/` importable; the deck builders are the
gate's own (`tests/test_rotational_symmetry_1029.py`), so a deck cannot
drift between the record and the test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from test_rotational_symmetry_1029 import WL, solver  # noqa: E402

from momwire import BSplineSolver  # noqa: E402

SPAN = (0.9, 1.0, 1.1)


def _sha(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _hex(z):
    z = complex(np.atleast_1d(np.asarray(z)).ravel()[0])
    return [z.real.hex(), z.imag.hex()]


def _free_dipole():
    """A deck the route never sees, on the fully batched swept path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=[np.array([(0.0, 0.0, -5.0), (0.0, 0.0, 5.0)])],
            n_per_edge_per_wire=[[21]],
            degree=2,
            wavelength=WL,
            wire_radius=0.001,
            feeds=[(0, 5.0, 1 + 0j)],
        )


def dense(path):
    out = {}
    for name, build in (
        ("free-dipole-batched", _free_dipole),
        ("screen4-flag-off-per-k", lambda: solver(4, rotational_symmetry=False)),
    ):
        s = build()
        ks = s.k * np.array(SPAN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = time.perf_counter()
            z = np.asarray(s.compute_impedance_swept(ks))
            dt = time.perf_counter() - t
        out[name] = {
            "batched": bool(s._swept_batched_available()),
            "shape": list(z.shape),
            "dtype": str(z.dtype),
            "sha": _sha(z),
            "z_hex": [_hex(zi) for zi in np.atleast_1d(z)],
            "seconds": round(dt, 3),
        }
        print(name, out[name]["sha"], out[name]["shape"], out[name]["dtype"])
    Path(path).write_text(json.dumps(out, indent=1) + "\n")


def route(path):
    rows = []
    for n in (4, 12):
        s = solver(n)
        d = solver(n, rotational_symmetry=False)
        ks = s.k * np.array(SPAN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = time.perf_counter()
            z_sw = np.asarray(s.compute_impedance_swept(ks))
            t_route = time.perf_counter() - t
            t = time.perf_counter()
            z_dense = np.asarray(d.compute_impedance_swept(ks))
            t_dense = time.perf_counter() - t
            per_k, per_k_fresh = [], []
            for kk in ks:
                s._set_k(float(kk))
                per_k.append(complex(np.atleast_1d(s.compute_impedance()[0])[0]))
                f = solver(n, wavelength=2.0 * np.pi / float(kk))
                per_k_fresh.append(complex(np.atleast_1d(f.compute_impedance()[0])[0]))
            s._set_k(float(ks[1]))
        z_sw1 = np.atleast_1d(z_sw).astype(np.complex128)
        rows.append(
            {
                "radials": n,
                "shape": list(z_sw.shape),
                "dtype": str(z_sw.dtype),
                "sha_route_swept": _sha(z_sw),
                "seconds_route": round(t_route, 3),
                "seconds_dense": round(t_dense, 3),
                "k": [float(x) for x in ks],
                "z_swept_hex": [_hex(z) for z in z_sw1],
                "z_per_k_hex": [_hex(z) for z in per_k],
                # 2a: the swept answer against the route's own per-frequency
                # compute_impedance, same solver object, k rebound the way the
                # loop rebinds it.
                "bit_identical_per_k": [
                    _hex(a) == _hex(b) for a, b in zip(z_sw1, per_k)
                ],
                "rel_per_k": [abs(a - b) / abs(b) for a, b in zip(z_sw1, per_k)],
                # 2a', the same against a solver CONSTRUCTED at each
                # wavelength — what a consumer does when it does not sweep.
                "rel_per_k_fresh_solver": [
                    abs(a - b) / abs(b) for a, b in zip(z_sw1, per_k_fresh)
                ],
                # 2b: against the dense swept path, per frequency.
                "rel_dense_swept": [
                    abs(a - b) / abs(b)
                    for a, b in zip(z_sw1, np.atleast_1d(z_dense).astype(complex))
                ],
                "copy_spread": s._rotational_copy_spread,
            }
        )
        print(json.dumps(rows[-1]))
    Path(path).write_text(json.dumps(rows, indent=1) + "\n")


def ak(path, radials=(4, 12)):
    """The consumer's own entry point, both ways round."""
    sys.path.insert(0, str(HERE))
    from feasibility import GROUND

    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine

    rows = []
    for n in radials:
        b = Builder()
        b.n_radials = n
        freqs = np.array([float(b.freq) * f for f in SPAN])
        got = {}
        # A COLD pass first, untimed. The Sommerfeld grid is built per
        # wavenumber and cached per process, so whichever engine sweeps first
        # pays for all three grids and the other reads them free — the two
        # timings below are only comparable once neither is the one paying.
        MomwireEngine(
            b, solver=BSplineSolver, solver_kwargs={"degree": 2}, ground=GROUND
        ).impedance_sweep(freqs)
        for name, kw in (
            ("dense", {"degree": 2}),
            ("route", {"degree": 2, "rotational_symmetry": True}),
        ):
            eng = MomwireEngine(
                b, solver=BSplineSolver, solver_kwargs=kw, ground=GROUND
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = time.perf_counter()
                z = np.asarray(eng.impedance_sweep(freqs))
                got[name] = (round(time.perf_counter() - t, 3), z)
        z_d, z_r = got["dense"][1], got["route"][1]
        rows.append(
            {
                "radials": n,
                "freqs_mhz": [float(f) for f in freqs],
                "shape_dense": list(z_d.shape),
                "shape_route": list(z_r.shape),
                "dtype": str(z_r.dtype),
                "seconds_dense": got["dense"][0],
                "seconds_route": got["route"][0],
                "speedup": round(got["dense"][0] / got["route"][0], 2),
                "rel": [abs(a - b) / abs(b) for a, b in zip(z_r.ravel(), z_d.ravel())],
            }
        )
        print(json.dumps(rows[-1]))
    Path(path).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense")
    ap.add_argument("--route")
    ap.add_argument("--ak")
    a = ap.parse_args()
    if a.dense:
        dense(a.dense)
    if a.route:
        route(a.route)
    if a.ak:
        ak(a.ak)
