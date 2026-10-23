"""The C++ sinusoidal front half against the Python one, deck by deck.

`src/momwire/_sinusoidal_front_core.cpp` copies what `SinusoidalSolver` derives
from its constructor arguments before any fill (momwire#1607 B2): the
constants, the junction normalisation, `_build_geometry` and
`_basis_coefs(geom, k)`.

The arguments come from the NEC-4.2 build (A2b). For each deck,
`nec4_build_compare.py`'s recorder captures every operating point's
constructor kwargs exactly as the slot builds them; here each is handed to
the REAL `SinusoidalSolver`, and its geometry and basis are dumped in the
format the core's `dump_front` writes. A point or structure the C++
declines is counted apart and not compared.

    python scripts/core_split/sinusoidal_front_compare.py DIR [DIR ...] [--show 3] [--out FILE]
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import warnings

import numpy as np

from momwire.deck._nec5 import ends_on_nx, split_structures
from momwire.eznec import _nec4
from momwire.portal import _portal
from momwire.sinusoidal import SinusoidalSolver

HERE = pathlib.Path(__file__).resolve().parent
SUFFIXES = {".nec", ".deck"}


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load("core_split_nec4_build_compare", "nec4_build_compare.py")
dc, sc = bc.dc, bc.sc


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, frozenset):
        return sorted(x)
    return x


def _front(kw: dict, p: str, out: list[str]) -> None:
    try:
        s = SinusoidalSolver(**kw)
        g = s._build_geometry()
        b = s._basis_coefs(g, s.k)
    except Exception as exc:  # noqa: BLE001 — the outcome IS the exception
        out.append(dc._error(p, exc))
        return
    for name in ("c", "freq", "omega", "k", "eta"):
        dc.dump(float(getattr(s, name)), f"{p}.{name}", out)
    dc.dump(s.junctions, f"{p}.junctions", out)
    dc.dump({k: _plain(v) for k, v in g.items()}, f"{p}.geometry", out)
    dc.dump({k: _plain(v) for k, v in b.items()}, f"{p}.basis", out)


def _points(text, echo, closing):
    """The structure's recorded constructor kwargs, and whether it went on."""
    with sc._stubbed() as seen:
        try:
            _nec4.render_structure(
                text, basis="sinusoidal", echo_text=echo, closing=closing
            )
        except sc._Served:
            served = dict(seen)
        except Exception:  # noqa: BLE001
            return [], False
        else:
            return [], False
    with bc._recording() as built, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = True
        try:
            solver = _portal.DeckSolver(served["deck"], basis="sinusoidal")
            for _index, group in served["plan"]:
                for freq in group.freqs_mhz:
                    solver.at(
                        freq, extended_kernel=group.ek, environment=group.environment
                    )
        except Exception:  # noqa: BLE001 — refused or raised: the build's gate
            ok = False
    return [dict(b.solver.recorded) for b in built], ok


def py_front(text: str) -> str:
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
        points, ok = _points(t, e, closing)
        recorded.append(points)
        if not ok:
            break
    out.append(f"front.structures : list {len(recorded)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s, points in enumerate(recorded):
            p = f"front.structures[{s}]"
            out.append(f"{p}.points : list {len(points)}")
            for j, kw in enumerate(points):
                _front(kw, f"{p}.points[{j}]", out)
    return "".join(line + "\n" for line in out)


def cxx_front(text: str, acc=None) -> str:
    return (acc or dc.accel()).sinusoidal_front_dump(text.encode("latin-1"))


def classify(dump: str) -> str:
    if " ? declined " in dump or dump.startswith("front ? declined"):
        return "declined"
    if " = float " not in dump and " ! " not in dump:
        return "empty"
    return "fronts"


def deck_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in SUFFIXES]


# The front is bit for bit on x86 Linux, which is what the standalone NEC-4.2
# executable ships on (momwire#1512). Elsewhere numpy's float64 sin and cos
# need not be libm's, which is what the core calls: on macOS arm64 the
# momwire#1651 arc probe p78 put one basis coefficient one ulp apart
# (3fe1a844feeaaca3 against ...aca2), every other field equal. Off Linux a
# float or complex field is therefore the same within `_OFF_LINUX_RTOL`, the
# NEC-5 seam's off-Linux bar (`seam_compare._OFF_LINUX_RTOL`); every other
# field, and every field on Linux, still compares byte for byte.
_OFF_LINUX_RTOL = 1e-11


def _hex_double(text: str) -> float:
    return float(np.frombuffer(bytes.fromhex(text)[::-1], dtype="<f8")[0])


def _within(x: str, y: str) -> bool:
    """Two dump lines that differ only in float or complex digits, each
    component within `_OFF_LINUX_RTOL` of the other."""
    (kx, _, vx), (ky, _, vy) = x.partition(" = "), y.partition(" = ")
    tx, ty = vx.split(), vy.split()
    if kx != ky or not tx or tx[0] != ty[0] or tx[0] not in ("float", "complex"):
        return False
    if len(tx) != len(ty):
        return False
    for hx, hy in zip(tx[1:], ty[1:], strict=True):
        a, b = _hex_double(hx), _hex_double(hy)
        if not (abs(a - b) <= _OFF_LINUX_RTOL * max(abs(a), abs(b))):
            return False
    return True


def _difference(py: str, cxx: str) -> str | None:
    diff = dc.first_difference(py, cxx)
    if diff is None or sys.platform.startswith("linux"):
        return diff
    lp, lc = py.splitlines(), cxx.splitlines()
    if len(lp) != len(lc):
        return diff
    for k, (x, y) in enumerate(zip(lp, lc, strict=True)):
        if x != y and not _within(x, y):
            return f"line {k}: python {x!r}\n          c++    {y!r}"
    return None


def compare(paths, out=None):
    acc = dc.accel()
    counts = {"fronts": 0, "empty": 0, "declined": 0, "mismatch": 0}
    mismatches: list[tuple[str, str]] = []
    for path in paths:
        text = path.read_bytes().decode("latin-1")
        cxx = cxx_front(text, acc)
        kind = classify(cxx)
        diff = None
        if kind != "declined":
            diff = _difference(py_front(text), cxx)
            if diff is not None:
                kind = "mismatch"
                mismatches.append((str(path), diff))
        counts[kind] += 1
        if out is not None:
            print(
                kind,
                path,
                "" if diff is None else diff.replace("\n", " | ")[:400],
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
