"""The C++ NEC-4.2 run half (sinusoidal basis) against the Python slot, deck
by deck.

`src/momwire/_nec4_run_core.cpp` with `_sinusoidal_run_core.cpp` copies
`eznec/_nec4.py`'s `solve` / `_run` over `SinusoidalSolver` (momwire#1607
B4a). Here each deck's printout from the C++ (`sinusoidal_nec4_printout`)
is held equal to `_nec4.render(text, basis="sinusoidal")`, byte for byte,
except the two lines that carry MEASURED time, which are masked by their
exact pattern and counted per deck (`mask`):

    "                        FILL=<%9.3f> SEC.,  FACTOR=<%9.3f> SEC."
    " RUN TIME =<%10.3f>"

A deck the C++ declines is counted apart, with its reason.

    python scripts/core_split/nec4_run_compare.py DIR [DIR ...] [--show 3] [--out FILE]
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.util
import os
import pathlib
import re
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SUFFIXES = {".nec", ".deck"}
BASIS = "sinusoidal"

# The two measured-time lines, exactly (`_printout._timing`, `_closing`).
FILL_LINE = re.compile(r"^ {24}FILL=[ 0-9.]{9} SEC\.,  FACTOR=[ 0-9.]{9} SEC\.$")
RUN_TIME_LINE = re.compile(r"^ RUN TIME =[ 0-9.]{10}$")


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pc = _load("core_split_nec4_printout_compare", "nec4_printout_compare.py")


def mask(text: str) -> tuple[str, int]:
    """`text` with each measured-time line's numbers blanked, and how many
    lines were masked."""
    out, n = [], 0
    for line in text.split("\n"):
        if FILL_LINE.match(line):
            out.append(" " * 24 + "FILL=<time> SEC.,  FACTOR=<time> SEC.")
            n += 1
        elif RUN_TIME_LINE.match(line):
            out.append(" RUN TIME =<time>")
            n += 1
        else:
            out.append(line)
    return "\n".join(out), n


def blas_addresses() -> list[int]:
    """numpy's ILP64 cblas_zgemm / zgemv / zdotu_sub / zgesv, SciPy's LP64
    zgetrf / zgetrs, and numpy's cblas_dgemm / dgemv / ddot."""
    import scipy

    def lib(pkg, pattern):
        here = os.path.dirname(pkg.__file__)
        found = glob.glob(os.path.join(here, "..", pattern)) or glob.glob(
            os.path.join(here, pattern)
        )
        return ctypes.CDLL(sorted(found)[0])

    npl = lib(np, "numpy.libs/libscipy_openblas64_*")
    spl = lib(scipy, "scipy.libs/libscipy_openblas*")
    names = (
        (npl, "scipy_cblas_zgemm64_"),
        (npl, "scipy_cblas_zgemv64_"),
        (npl, "scipy_cblas_zdotu_sub64_"),
        (npl, "scipy_zgesv_64_"),
        (spl, "scipy_zgetrf_"),
        (spl, "scipy_zgetrs_"),
        (npl, "scipy_cblas_dgemm64_"),
        (npl, "scipy_cblas_dgemv64_"),
        (npl, "scipy_cblas_ddot64_"),
    )
    return [ctypes.cast(getattr(lb, n), ctypes.c_void_p).value for lb, n in names]


def py_printout(text: str) -> str:
    from momwire.eznec import _nec4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _nec4.render(text, basis=BASIS)


def cxx_printout(text: str, acc=None, blas=None) -> tuple[bool, str]:
    acc = acc or pc.accel()
    version, variant, marker = pc.stamp()
    return acc.sinusoidal_nec4_printout(
        text.encode("latin-1"), blas or blas_addresses(), version, variant, marker
    )


def first_difference(a: str, b: str) -> str | None:
    x, y = a.split("\n"), b.split("\n")
    for k, (p, q) in enumerate(zip(x, y, strict=False)):
        if p != q:
            return f"line {k + 1}: python {p!r} | c++ {q!r}"
    if len(x) != len(y):
        return f"python {len(x)} lines, c++ {len(y)}"
    return None


def deck_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in SUFFIXES]


def compare(paths, out=None):
    acc = pc.accel()
    blas = blas_addresses()
    counts = {"printed": 0, "declined": 0, "mismatch": 0, "py_raised": 0}
    mismatches: list[tuple[str, str]] = []
    masked_total = 0
    for path in paths:
        text = path.read_bytes().decode("latin-1")
        ok, cxx = cxx_printout(text, acc, blas)
        detail = ""
        if not ok:
            kind, detail = "declined", cxx
        else:
            try:
                py = py_printout(text)
            except Exception as exc:  # noqa: BLE001 — recorded, the run goes on
                kind, detail = "py_raised", f"{type(exc).__name__}: {exc}"[:300]
            else:
                a, na = mask(py)
                b, nb = mask(cxx)
                diff = first_difference(a, b)
                if diff is None and na != nb:
                    diff = f"masked {na} python lines, {nb} c++"
                if diff is None:
                    kind, detail = "printed", f"masked={na}"
                    masked_total += na
                else:
                    kind, detail = "mismatch", diff
                    mismatches.append((str(path), diff))
        counts[kind] += 1
        if out is not None:
            print(kind, path, detail.replace("\n", " | ")[:400], file=out, flush=True)
    counts["masked_lines"] = masked_total
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
            counts, mismatches = compare(deck_files(root), out)
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
