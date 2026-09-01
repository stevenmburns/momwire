"""Profile `SinusoidalGalerkinSolver` at DEFAULT settings (momwire#194 step 1).

Phase-attributed wall-clock and peak RSS across an N ladder, with the
point-matched `SinusoidalSolver` as the baseline column (the issue's "the
Galerkin fill is ~n_qp× the point-matched fill by construction" claim,
measured). Each rung runs in its OWN child process so peak RSS is per-rung,
and every child runs under a hard address-space cap (default 24 GiB) so an
over-budget rung dies with a recorded MemoryError instead of an OOM kill —
the M6 sweep was OOM-killed at 2379 segments on a 32 GiB box, and that
ceiling is part of what this profile is for.

Phases (nested; `total` contains `assemble`, which contains the rest):

  setup        `_basis_coefs` + `_test_context`
  kernel_far   the far fill's field evaluation. On the numpy path that is
               `_field_components_bcast` — the Eqs 76-79 tables at
               (N·n_qp_test, N) with the n_qp_const source-quadrature axis on
               top, i.e. the O(N²·nq) workspace. With the C++ accelerator it
               is `_far_fill_accel`, which fuses the kernel WITH the reduction
               (#194 step 2), so `reduce` goes to zero and this column is the
               whole far fill.
  reduce       the per-quad-node test-integration sum
               (`_tested_contrib_rows`, on the numpy far fill and on the
               Sommerfeld remainder alike)
  near         `_apply_near_correction` total (includes its kernel calls)
  assemble     `_assemble_Z` total
  total        `compute_impedance` total (assemble + factor/solve + readout)

`--no-accel` forces the pure-Python far fill, which is what the before/after
comparison for #194 step 2 is made of: same binary, same ladder, one flag.

Geometry: the M1-M3 validation dipole (0.962λ/2, a = 0.5 mm), the same
first-party geometry the M6 harness's feed-model column uses. The cost
structure under test is per-N kernel/workspace arithmetic, which is
geometry-agnostic; junction-heavy geometry changes only the O(N) near-pair
count.

Running
-------
    python scripts/profile_sinusoidal_galerkin.py                 # default ladder
    python scripts/profile_sinusoidal_galerkin.py --nsegs 101 401
    python scripts/profile_sinusoidal_galerkin.py --cap-gib 24
    python scripts/profile_sinusoidal_galerkin.py --no-accel      # numpy far fill
    python scripts/profile_sinusoidal_galerkin.py --out prof.json
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from momwire import SinusoidalGalerkinSolver, SinusoidalSolver

DEFAULT_LADDER = (101, 201, 401, 801, 1201, 1601, 2001, 2401)


def dipole_kwargs(n: int) -> dict:
    hd = 0.962 * 22 / 4
    return dict(
        wires=[np.array([[0, 0, -hd], [0, 0, hd]], dtype=float)],
        n_per_edge_per_wire=[[n]],
        feed_wire_index=0,
        feed_arclength=hd,
        wavelength=22,
        wire_radius=0.0005,
    )


# ---------------------------------------------------------------------------
# instrumented solver — timing wrappers only, no behavior change
# ---------------------------------------------------------------------------
class _ProfiledGalerkin(SinusoidalGalerkinSolver):
    """Wall-clock per phase into `self.prof`; math untouched."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prof: dict[str, float] = {}
        self._in_near = False
        self.n_near_pairs = 0

    def _timed(self, key, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.prof[key] = self.prof.get(key, 0.0) + (time.perf_counter() - t0)
        return out

    def _basis_coefs(self, *a, **kw):
        return self._timed("setup", super()._basis_coefs, *a, **kw)

    def _test_context(self, *a, **kw):
        return self._timed("setup", super()._test_context, *a, **kw)

    def _field_components_bcast(self, *a, **kw):
        key = "kernel_near" if self._in_near else "kernel_far"
        return self._timed(key, super()._field_components_bcast, *a, **kw)

    def _far_fill_accel(self, *a, **kw):
        # The fused C++ far fill: kernel AND reduction, so it lands in
        # kernel_far and `reduce` stays empty on this path.
        return self._timed("kernel_far", super()._far_fill_accel, *a, **kw)

    def _tested_contrib_rows(self, *a, **kw):
        # An instance method shadowing the base staticmethod — every reduction
        # in the fill calls it directly (there is no whole-matrix entry point
        # since #332), so without it the reduce column reads zero.
        return self._timed(
            "reduce", SinusoidalGalerkinSolver._tested_contrib_rows, *a, **kw
        )

    def _near_pairs(self, *a, **kw):
        mm, nn = super()._near_pairs(*a, **kw)
        self.n_near_pairs += int(mm.size)
        return mm, nn

    def _apply_near_correction(self, *a, **kw):
        self._in_near = True
        try:
            return self._timed("near", super()._apply_near_correction, *a, **kw)
        finally:
            self._in_near = False

    def _assemble_Z(self, *a, **kw):
        return self._timed("assemble", super()._assemble_Z, *a, **kw)

    def compute_impedance(self):
        return self._timed("total", super().compute_impedance)


def run_one(n: int, no_accel: bool = False) -> dict:
    kw = dipole_kwargs(n)
    if no_accel:
        from momwire import sinusoidal_galerkin as _sg

        _sg._HAVE_GALERKIN_FAR_FILL = False

    t0 = time.perf_counter()
    z_coll, _ = SinusoidalSolver(**kw).compute_impedance()
    coll_total = time.perf_counter() - t0

    s = _ProfiledGalerkin(**kw)
    z_gal, _ = s.compute_impedance()

    zg = complex(np.atleast_1d(z_gal)[0])
    zc = complex(np.atleast_1d(z_coll)[0])
    return {
        "n": n,
        "z_gal": [zg.real, zg.imag],
        "z_coll": [zc.real, zc.imag],
        "coll_total": coll_total,
        "near_pairs": s.n_near_pairs,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        **{k: round(v, 4) for k, v in s.prof.items()},
    }


# ---------------------------------------------------------------------------
# parent / child protocol: one child per rung, hard AS cap in the child
# ---------------------------------------------------------------------------
def child_main(n: int, cap_gib: float, no_accel: bool = False) -> None:
    cap = int(cap_gib * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    try:
        row = run_one(n, no_accel=no_accel)
    except MemoryError:
        row = {
            "n": n,
            "oom": True,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        }
    print(json.dumps(row))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--nsegs", nargs="+", type=int, default=list(DEFAULT_LADDER))
    ap.add_argument("--cap-gib", type=float, default=24.0)
    ap.add_argument(
        "--no-accel",
        action="store_true",
        help="force the pure-Python far fill (the #194 before column)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.single is not None:
        child_main(args.single, args.cap_gib, no_accel=args.no_accel)
        return

    cols = (
        "coll_total total assemble kernel_far reduce near kernel_near "
        "setup near_pairs peak_rss_gib"
    ).split()
    print(
        f"cap: {args.cap_gib} GiB (RLIMIT_AS, per child)   ladder: {args.nsegs}"
        f"   far fill: {'numpy (forced)' if args.no_accel else 'accelerated'}"
    )
    print(f"{'N':>6} " + " ".join(f"{c:>12}" for c in cols))
    rows = []
    for n in args.nsegs:
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single",
                str(n),
                "--cap-gib",
                str(args.cap_gib),
            ]
            + (["--no-accel"] if args.no_accel else []),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"{n:>6}  child died rc={proc.returncode}: {proc.stderr[-300:]}")
            rows.append({"n": n, "died": True})
            continue
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(row)
        if row.get("oom"):
            print(f"{n:>6}  MemoryError under cap (peak {row['peak_rss_gib']:.1f} GiB)")
            continue
        print(
            f"{n:>6} "
            + " ".join(
                (
                    f"{row.get(c, 0):>12.2f}"
                    if c != "near_pairs"
                    else f"{row.get(c, 0):>12d}"
                )
                for c in cols
            )
            + f"   Zgal={row['z_gal'][0]:.2f}{row['z_gal'][1]:+.2f}j"
        )

    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
