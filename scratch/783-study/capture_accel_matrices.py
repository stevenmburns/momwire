"""momwire#783 unit 1: capture accelerated outputs, for the bit-identity proof.

The rename is a pure token substitution, so the acceptance bar is that the
extension is BIT-IDENTICAL afterwards -- proved by rebuilding and comparing,
not by reasoning about sed (the #762 protocol).

One capture per accelerated translation unit that carries the macros, so a
typo confined to any single file would show:

    _accel_bspline.cpp    the B-spline off-edge fill
    _accel_razor.cpp      razor's assemble
    _accel_sinusoidal.cpp the sinusoidal fill
    _accel_somm.cpp       the Sommerfeld remainder projection

Run before the rename with `before`, after with `after`; the second run
compares and exits non-zero on any difference.

    python scratch/783-study/capture_accel_matrices.py before
    ... rename, make build ...
    python scratch/783-study/capture_accel_matrices.py after
"""

import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "accel_capture.npz"
WL = 20.0


def _deck(cls_name):
    """A small deck each family solves, exercising its own fill."""
    # The two wires must SHARE the junction point, or `normalize_junctions`
    # refuses before any fill runs.
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]),
        np.array([(0.0, 0.0, 0.0), (3.0, 0.0, 1.0)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[15], [9]],
        junctions=[[(0, "start"), (1, "start")]] if cls_name != "PulseSolver" else None,
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL,
        wire_radius=1e-3,
    )


def capture():
    """One (Z, current vector) per family, via the PUBLIC solve.

    The currents are captured rather than a private assembly matrix: they sit
    downstream of the entire fill, so any change in any accelerated kernel
    moves them, and the capture does not depend on each family's internal
    method names (which differ).
    """
    from momwire import (
        BSplineSolver,
        RazorSolver,
        SinusoidalGalerkinSolver,
    )

    def kw(extra=None):
        d = {k: v for k, v in _deck("x").items() if v is not None}
        d.update(extra or {})
        return d

    out = {}
    for name, cls, extra in (
        ("bspline", BSplineSolver, {"degree": 2}),
        ("sinusoidal", SinusoidalGalerkinSolver, {}),
        ("razor", RazorSolver, {}),
        (
            "somm",
            BSplineSolver,
            {
                "degree": 2,
                "ground_z": -3.0,
                "ground_eps": (13.0, 0.005),
                "ground_model": "sommerfeld",
            },
        ),
    ):
        z, currents = cls(**kw(extra)).compute_impedance()
        out[f"{name}_Z"] = np.asarray([complex(z)], dtype=np.complex128)
        out[f"{name}_I"] = np.asarray(currents, dtype=np.complex128)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "before"
    got = capture()
    for k, v in got.items():
        print(f"  {k:16s} shape {v.shape}  |Z|sum {np.abs(v).sum():.12e}")
    if mode == "before":
        np.savez(OUT, **got)
        print(f"saved {OUT}")
        return 0
    ref = np.load(OUT)
    bad = 0
    for k, v in got.items():
        same = np.array_equal(ref[k], v)
        print(f"  {k:16s} array_equal -> {same}")
        if not same:
            bad += 1
            d = np.abs(ref[k] - v)
            print(
                f"      max |diff| {d.max():.3e} at {np.unravel_index(d.argmax(), d.shape)}"
            )
    print("BIT-IDENTICAL" if not bad else f"{bad} MATRIX(ES) DIFFER")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
