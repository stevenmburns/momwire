"""momwire#1154 probe 5: A/B against main, bit for bit.

Run once per source tree (PYTHONPATH picks it); writes every result to an
.npz named by argv[1]. `compare` mode loads two and reports array_equal per
case. The last two cases are CONTROLS that must differ (a buried jacket),
so an A/B that reports everything equal has provably run two trees.
"""

import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")

WL = 299792458.0 / 7.0e6
SOIL = (13.0, 0.005)
J = dict(insulation_radius=1.8e-3, insulation_eps_r=3.5)


def dip(z0, length=5.0, n=21, **kw):
    pts = np.array([(-0.5 * length, 0.0, z0), (0.5 * length, 0.0, z0)])
    return dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, (n // 2 + 0.5) / n * length, 1 + 0j)],
        wavelength=WL,
        wire_radius=1e-3,
        **kw,
    )


def cases():
    import momwire
    from momwire.bspline import BSplineSolver
    from momwire.razor import RazorSolver
    from momwire.sinusoidal import SinusoidalSolver
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

    # The deck builders come from the BRANCH's tests (identical on main); the
    # solver code comes from whichever tree PYTHONPATH selected.
    sys.path.insert(0, os.environ["MW_TESTS"])
    print("momwire from", momwire.__file__, flush=True)
    from test_razor_crossing_loading_1149 import crossing

    som = dict(ground_z=0.0, ground_eps=SOIL, ground_model="sommerfeld")
    refl = dict(ground_z=0.0, ground_eps=SOIL)
    pec = dict(ground_z=0.0)
    all4 = (BSplineSolver, RazorSolver, SinusoidalSolver, SinusoidalGalerkinSolver)
    out = []
    for cls in all4:
        out.append((f"free/{cls.__name__}", cls, dip(0.0, **J)))
        out.append((f"pec/{cls.__name__}", cls, dip(3.0, **J, **pec)))
        out.append((f"refl/{cls.__name__}", cls, dip(3.0, **J, **refl)))
    for cls in (BSplineSolver, RazorSolver, SinusoidalGalerkinSolver):
        out.append((f"somm-above/{cls.__name__}", cls, dip(3.0, **J, **som)))
    # a jacketed wire RESTING on the soil (h = b): above by the media labels
    out.append(("surface-h=b/BSplineSolver", BSplineSolver, dip(1.8e-3, **J, **som)))
    for cls in (BSplineSolver, RazorSolver):
        d = crossing(1)
        d["insulation_radius"] = [np.nan, 2.0 * np.max(d["wire_radius"])]
        d["insulation_eps_r"] = [np.nan, 3.0]
        out.append((f"crossing-above-jacket/{cls.__name__}", cls, d))
        out.append(
            (
                f"buried-bare-sigma/{cls.__name__}",
                cls,
                dip(-0.5, wire_conductivity=3.5e7, **som),
            )
        )
    for cls in (BSplineSolver, RazorSolver):
        out.append(
            (f"CONTROL-buried-jacket/{cls.__name__}", cls, dip(-0.5, **J, **som))
        )
    return out


def run(path):
    from momwire.razor import RazorSolver

    res = {}
    for name, cls, d in cases():
        d = dict(d)
        extra = {}
        if cls is RazorSolver:
            d.pop("junctions", None)
            extra = dict(nec5_quadrature=True)
        s = cls(**d, **extra)
        z = np.asarray(s.compute_impedance()[0], dtype=np.complex128)
        res[name] = z
        if name.startswith(("free/B", "somm-above/B", "free/R")):
            ks = s.k * np.array([0.9, 1.0, 1.1])
            res[name + "/swept"] = np.asarray(
                cls(**d, **extra).compute_impedance_swept(ks), dtype=np.complex128
            )
        print(name, z, flush=True)
    np.savez(path, **res)


def compare(a, b):
    A, B = np.load(a), np.load(b)
    bad = 0
    for k in A.files:
        eq = np.array_equal(A[k], B[k])
        ctrl = k.startswith("CONTROL")
        ok = (not eq) if ctrl else eq
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {'differs' if not eq else 'equal  '} {k}")
    print("FAILURES:", bad)
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    run(sys.argv[1])
