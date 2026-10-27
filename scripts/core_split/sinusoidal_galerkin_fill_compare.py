"""The C++ Sinusoidal-Galerkin mixed-potential fill and solve against the
Python one, deck by deck.

`src/momwire/_sinusoidal_galerkin_fill_core.cpp` copies
`SinusoidalGalerkinSolver.compute_port_solution` on the default
`fill="mixed-potential"` route in free space (momwire#1718 S2): `_assemble_Z_mp`'s
windows (the tiered pair moments, the near pairs, the assembler),
`_drive_columns`, `_solve_in_place` and `_port_currents` into Y.

The operating points come from the NEC-4.2 build, recorded exactly as
`sinusoidal_front_compare.py` records them (basis "sinusoidal"); each point's
constructor kwargs, less the point-matched lane's `junction_rule`, are handed
to the REAL `SinusoidalGalerkinSolver`, its `compute_port_solution()` is run
with `_assemble_Z`'s G captured, and G, the coefficients and Y are dumped in
the format the core's `dump_fill` writes: whole up to 64 columns (or `whole`),
by row digest beyond. A deck the C++ declines anywhere is counted apart and
not compared.

    python scripts/core_split/sinusoidal_galerkin_fill_compare.py DIR [DIR ...] [--show 3] [--out FILE]
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.util
import os
import pathlib
import sys
import warnings

import numpy as np

from momwire.deck._nec5 import ends_on_nx, split_structures
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

HERE = pathlib.Path(__file__).resolve().parent
SUFFIXES = {".nec", ".deck"}
WHOLE = 64  # `kWhole` in the core


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fc = _load("core_split_sinusoidal_front_compare", "sinusoidal_front_compare.py")
dc = fc.dc

_P = np.uint64(0x100000001B3)
_MIX = np.uint64(0xBF58476D1CE4E5B9)


def _digest(rows: np.ndarray) -> list[str]:
    """Each row's digest, as the core's `digest`: its float64 words w,
    x = (w ^ w >> 29) * MIX, x ^= x >> 32, h = sum x_i P^(i+1) mod 2^64."""
    w = np.ascontiguousarray(rows, dtype=np.complex128).view(np.uint64)
    pw = np.cumprod(np.full(w.shape[1], _P, dtype=np.uint64), dtype=np.uint64)
    out = []
    for r in w:
        x = r ^ (r >> np.uint64(29))
        x = x * _MIX
        x ^= x >> np.uint64(32)
        out.append(f"{int(np.sum(x * pw, dtype=np.uint64)):016x}")
    return out


def _matrix(m: np.ndarray, p: str, out: list[str], whole: int = WHOLE) -> None:
    rows, cols = m.shape
    out.append(f"{p} : shape {rows} {cols}")
    if cols <= whole:
        for i in range(rows):
            for j in range(cols):
                v = complex(m[i, j])
                out.append(f"{p}[{i}][{j}] = complex {dc._f(v.real)} {dc._f(v.imag)}")
        return
    for i, d in enumerate(_digest(m)):
        out.append(f"{p}[{i}] = digest {d}")


def _fill(
    kw: dict, p: str, out: list[str], whole: int = WHOLE, g_only: bool = False
) -> None:
    kw = {k: v for k, v in kw.items() if k != "junction_rule"}
    try:
        s = SinusoidalGalerkinSolver(**kw)
        s._build_geometry()
    except Exception as exc:  # noqa: BLE001 — the outcome IS the exception
        out.append(dc._error(p, exc))
        return
    seen = {}
    orig = s._assemble_Z

    def capture(*a, **k):
        G, view = orig(*a, **k)
        seen["G"] = np.array(G, copy=True)
        return G, view

    s._assemble_Z = capture
    try:
        sol = s.compute_port_solution()
    except Exception as exc:  # noqa: BLE001
        out.append(dc._error(p, exc))
        return
    _matrix(seen["G"], f"{p}.G", out, whole)
    if g_only:
        return
    _matrix(np.asarray(sol.coeffs), f"{p}.coeffs", out, whole)
    _matrix(np.asarray(sol.y), f"{p}.y", out, whole)


def py_fill(text: str, whole: int = WHOLE, g_only: bool = False) -> str:
    out: list[str] = []
    structures = split_structures(text)
    items = (
        [(text, text, True)]
        if structures is None
        else [
            (s.deck_text, s.echo_text, k == len(structures) - 1 and not ends_on_nx(s))
            for k, s in enumerate(structures)
        ]
    )
    recorded = []
    for t, e, closing in items:
        points, ok = fc._points(t, e, closing)
        recorded.append(points)
        if not ok:
            break
    out.append(f"fill.structures : list {len(recorded)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s, points in enumerate(recorded):
            p = f"fill.structures[{s}]"
            out.append(f"{p}.points : list {len(points)}")
            for j, kw in enumerate(points):
                _fill(kw, f"{p}.points[{j}]", out, whole, g_only)
    return "".join(line + "\n" for line in out)


def blas_addresses() -> list[int]:
    """numpy's ILP64 cblas_zgemm / zgemv / zdotu_sub, SciPy's LP64 zgetrf /
    zgetrs, and numpy's cblas_dgemv / ddot, as `sinusoidal_galerkin_fill_dump`
    takes them."""
    import scipy

    def lib(pkg, pattern):
        found = glob.glob(
            os.path.join(os.path.dirname(pkg.__file__), "..", pattern)
        ) or glob.glob(os.path.join(os.path.dirname(pkg.__file__), pattern))
        return ctypes.CDLL(sorted(found)[0])

    npl = lib(np, "numpy.libs/libscipy_openblas64_*")
    spl = lib(scipy, "scipy.libs/libscipy_openblas*")
    names = (
        (npl, "scipy_cblas_zgemm64_"),
        (npl, "scipy_cblas_zgemv64_"),
        (npl, "scipy_cblas_zdotu_sub64_"),
        (spl, "scipy_zgetrf_"),
        (spl, "scipy_zgetrs_"),
        (npl, "scipy_cblas_dgemv64_"),
        (npl, "scipy_cblas_ddot64_"),
    )
    return [ctypes.cast(getattr(lb, n), ctypes.c_void_p).value for lb, n in names]


def cxx_fill(
    text: str, acc=None, blas=None, whole: int = WHOLE, g_only: bool = False
) -> str:
    """`g_only`: the fill alone (G), which takes no BLAS, so it runs where
    numpy bundles no OpenBLAS (macOS's Accelerate build)."""
    if g_only:
        blas = []
    elif blas is None:
        blas = blas_addresses()
    return (acc or dc.accel()).sinusoidal_galerkin_fill_dump(
        text.encode("latin-1"), blas, whole
    )


def classify(dump: str) -> str:
    if " ? declined " in dump or dump.startswith("fill ? declined"):
        return "declined"
    if " : shape " not in dump and " ! " not in dump:
        return "empty"
    return "fills"


def compare(paths, out=None):
    acc = dc.accel()
    blas = blas_addresses()
    counts = {"fills": 0, "empty": 0, "declined": 0, "mismatch": 0}
    mismatches: list[tuple[str, str]] = []
    for path in paths:
        text = path.read_bytes().decode("latin-1")
        cxx = cxx_fill(text, acc, blas)
        kind = classify(cxx)
        diff = None
        why = ""
        if kind == "declined":
            why = next(
                (
                    ln
                    for ln in cxx.splitlines()
                    if " ? declined " in ln or ln.startswith("fill ?")
                ),
                "",
            )
            why = bytes.fromhex(why.rsplit(" ", 1)[-1]).decode("utf-8", "replace")
        else:
            diff = dc.first_difference(py_fill(text), cxx)
            if diff is not None:
                kind = "mismatch"
                mismatches.append((str(path), diff))
        counts[kind] += 1
        if out is not None:
            print(
                kind,
                path,
                why if diff is None else diff.replace("\n", " | ")[:400],
                file=out,
                flush=True,
            )
    return counts, mismatches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args(argv)
    out = open(args.out, "w") if args.out else None  # noqa: SIM115 — closed below
    bad = 0
    try:
        for root in args.dirs:
            counts, mismatches = compare(fc.deck_files(root), out)
            print(f"{root}: {counts}", flush=True)
            for label, diff in mismatches[: args.show]:
                print(f"  MISMATCH {label}\n{diff}", flush=True)
            bad += counts["mismatch"]
    finally:
        if out is not None:
            out.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
