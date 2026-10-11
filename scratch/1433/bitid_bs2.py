"""PR A gate (momwire#1433): bs2 bit-identical across the exact-kernel
plumbing extraction, kernel off / True / "auto".

Run the SAME script under two trees (origin/main and the branch) and diff the
JSON lines. Each record carries the Z bytes' sha256 (compute_impedance and
the solved coefficients), the resolved `exact_kernel`, and how many times
`_add_exact_kernel_correction` actually ran -- instrumented, so a case that
claims the kernel engaged is known to have taken the correction route.

Usage: python -I bitid_bs2.py [quick|full] > out.jsonl
"""

import hashlib
import json
import pathlib
import sys
import traceback
import warnings

import numpy as np

import momwire
from momwire import BSplineSolver
from momwire.bspline import BSplineSolver as _B

HERE = pathlib.Path(__file__).resolve().parent
TESTS = HERE.parent.parent / "tests"
sys.path.insert(0, str(TESTS))
import _catalog_fixture as cat  # noqa: E402

CALLS = {"corr": 0}
_orig = _B._add_exact_kernel_correction


def _counted(self, *a, **k):
    CALLS["corr"] += 1
    return _orig(self, *a, **k)


_B._add_exact_kernel_correction = _counted


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


L = 0.47
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]
BENT = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])]
HALVES = [
    np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, -0.002)]),
    np.array([(0.0, 0.0, 0.002), (0.0, 0.0, L / 2)]),
]
TEE = [
    np.array([(0.0, 0.0, -0.2), (0.0, 0.0, 0.2)]),
    np.array([(0.0, 0.0, 0.0), (0.15, 0.0, 0.0)]),
]


def cases(level):
    out = []
    for kern in (False, True, "auto"):
        for n, a in ((19, 0.0005), (19, 0.00425), (41, 0.00425), (81, 0.00425)):
            out.append(
                (
                    f"dipole n{n} a{a}",
                    kern,
                    dict(
                        wires=DIPOLE,
                        n_per_edge_per_wire=[[n]],
                        wavelength=1.0,
                        wire_radius=a,
                    ),
                )
            )
        out.append(
            (
                "dipole EK n41",
                kern,
                dict(
                    wires=DIPOLE,
                    n_per_edge_per_wire=[[41]],
                    wavelength=1.0,
                    wire_radius=0.00425,
                    extended_kernel=True,
                ),
            )
        )
        out.append(
            (
                "bent n40",
                kern,
                dict(
                    wires=BENT,
                    n_per_edge_per_wire=[[20, 20]],
                    wavelength=1.0,
                    wire_radius=0.006,
                ),
            )
        )
        out.append(
            (
                "tee",
                kern,
                dict(
                    wires=TEE,
                    n_per_edge_per_wire=[[40], [15]],
                    wavelength=1.0,
                    wire_radius=0.005,
                ),
            )
        )
        out.append(
            (
                "two radii",
                kern,
                dict(
                    wires=HALVES,
                    n_per_edge_per_wire=[[40], [40]],
                    wavelength=1.0,
                    wire_radius=[0.006, 0.003],
                    feeds=[(0, None, 1.0)],
                ),
            )
        )
        out.append(
            (
                "endport",
                kern,
                dict(
                    wires=HALVES,
                    n_per_edge_per_wire=[[60], [60]],
                    wavelength=1.0,
                    wire_radius=0.00425,
                    feeds=[(0, (L - 0.004) / 4, 0j)],
                    junctions=[[(0, "end")], [(1, "start")]],
                    junction_ports=[(0, 0j), (1, 0j)],
                ),
            )
        )
        out.append(
            (
                "pec vertical",
                kern,
                dict(
                    wires=[np.array([(0, 0, 0.05), (0, 0, 0.05 + L)])],
                    n_per_edge_per_wire=[[60]],
                    wavelength=1.0,
                    wire_radius=0.00425,
                    ground_z=0.0,
                ),
            )
        )
        out.append(
            (
                "refl-coef",
                kern,
                dict(
                    wires=[np.array([(-L / 2, 0, 0.3), (L / 2, 0, 0.3)])],
                    n_per_edge_per_wire=[[60]],
                    wavelength=1.0,
                    wire_radius=0.00425,
                    ground_z=0.0,
                    ground_eps=(13.0, 0.005),
                ),
            )
        )
        out.append(
            (
                "buried",
                kern,
                dict(
                    wires=[np.array([(-L / 2, 0, -0.3), (L / 2, 0, -0.3)])],
                    n_per_edge_per_wire=[[60]],
                    wavelength=1.0,
                    wire_radius=0.00425,
                    ground_z=0.0,
                    ground_eps=(13.0, 0.005),
                    ground_model="sommerfeld",
                ),
            )
        )
        out.append(
            (
                "enrich",
                kern,
                dict(
                    wires=DIPOLE,
                    n_per_edge_per_wire=[[41]],
                    wavelength=1.0,
                    wire_radius=0.00425,
                    use_singular_enrichment=True,
                ),
            )
        )
        out.append(
            (
                "degree3",
                kern,
                dict(
                    wires=DIPOLE,
                    n_per_edge_per_wire=[[41]],
                    wavelength=1.0,
                    wire_radius=0.00425,
                    degree=3,
                ),
            )
        )
        if level == "full":
            for d in cat.available():
                try:
                    kw = cat.catalog_cell(d)
                except KeyError:
                    continue
                out.append((f"catalog {d}", kern, kw))
            out.append(
                (
                    "dipole n801 fat",
                    kern,
                    dict(
                        wires=DIPOLE,
                        n_per_edge_per_wire=[[801]],
                        wavelength=1.0,
                        wire_radius=0.00425,
                    ),
                )
            )
    return out


def run(label, kern, kw):
    kw = dict(kw)
    kw = {k: v for k, v in kw.items() if v is not None}
    rec = {"case": label, "kernel": repr(kern)}
    CALLS["corr"] = 0
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = BSplineSolver(exact_kernel=kern, **kw)
            rec["resolved"] = repr(s.exact_kernel)
            if "junction_ports" in kw:
                z = np.asarray(s.compute_y_matrix(), dtype=complex).ravel()
            else:
                res = s.compute_impedance()
                parts = res if isinstance(res, tuple) else (res,)
                z = np.concatenate(
                    [np.asarray(p, dtype=complex).ravel() for p in parts]
                )
        rec["Z"] = sha(z)
        rec["z0"] = [z[0].real, z[0].imag]
        rec["warn"] = sorted({f"{type(x.message).__name__}: {x.message}" for x in w})
    except Exception as exc:  # noqa: BLE001 -- the gate records refusals
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["corr_calls"] = CALLS["corr"]
    rec["exact_after"] = repr(getattr(locals().get("s"), "exact_kernel", None))
    return rec


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "quick"
    for label, kern, kw in cases(level):
        print(json.dumps(run(label, kern, kw)), flush=True)
    # Swept: the per-k loop.
    for kern in (False, True, "auto"):
        CALLS["corr"] = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = BSplineSolver(
                exact_kernel=kern,
                wires=DIPOLE,
                n_per_edge_per_wire=[[41]],
                wavelength=1.0,
                wire_radius=0.00425,
            )
            zs = np.asarray(s.compute_impedance_swept(np.array([5.9, 6.28, 6.6])))
        print(
            json.dumps(
                {
                    "case": "swept",
                    "kernel": repr(kern),
                    "Z": sha(zs),
                    "corr_calls": CALLS["corr"],
                }
            ),
            flush=True,
        )
    print(json.dumps({"momwire": momwire.__file__}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
