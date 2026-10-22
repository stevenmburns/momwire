"""The C++ NEC-4.2 build against the Python one, deck by deck.

`src/momwire/_nec4_build_core.cpp` copies what the NEC-4.2 slot does
between its seam front (`nec4_seam_compare.py`, momwire#1607 A2) and the
fill (momwire#1607 A2b). That is `portal.DeckSolver`'s construction and every
operating point `solve` asks it for, which covers:
- `deck._solver.prepare_mesh`: the union port set and `_polylines.to_polylines`,
  the chaining of the model's wires into the polylines a solver takes, with
  the junctions, the plane split and the ports placed on them;
- `build_solver` / `construct_solver`: the constructor arguments, the ground
  and wire-loading kwargs, the extended kernel as the dialect's default and
  its fallback or partial advisories, and the port plan in the solver's space;
- DeckSolver's NEC-facing tables: the union `(tag, seg)` ports, the global
  segments, the feed and load bridges, `port_by_segment`, current ports, and
  each fill's deck-sign congruence (`_port_signs`).

This module writes the Python side of the comparison format the core's
`dump_build` writes. Per structure the seam front runs as
`nec4_seam_compare.py` runs it. A served structure's DeckSolver is then
constructed for real, with three things stubbed:
- the solver class records its constructor's keyword arguments instead of
  filling;
- `DeckSolver._entry` records the built solver instead of factoring;
- `solve`'s run loop is restated as `at(freq, ek, environment)` per run and
  frequency in the plan.

The exceptions `solve` turns into a refusal (PortalError, ValueError,
NotImplementedError, LinAlgError) are refusals here too.

    python scripts/core_split/nec4_build_compare.py DIR [DIR ...] \\
        [--basis sinusoidal|bspline|both] [--driver PATH] [--show 5]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import pathlib
import subprocess
import sys
import types
import warnings

import numpy as np

from momwire.deck import _solver as deck_solver
from momwire.deck._nec5 import ends_on_nx, split_structures
from momwire.eznec import _nec4
from momwire.portal import _portal

HERE = pathlib.Path(__file__).resolve().parent
SUFFIXES = {".nec", ".deck"}
BASES = ("sinusoidal", "bspline")


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dc = _load("core_split_deck_compare", "deck_compare.py")
sc = _load("core_split_nec4_seam_compare", "nec4_seam_compare.py")


def _plain(value):
    """Arrays as lists (the comparison format has no ndarray spelling),
    recursively through the containers a constructor argument is built of."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (list, tuple)):
        out = [_plain(v) for v in value]
        return tuple(out) if isinstance(value, tuple) else out
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


@contextlib.contextmanager
def _recording():
    """The roster's solver classes record their constructor's kwargs; the
    DeckSolver records each built solver instead of filling it."""
    built: list = []
    saved_entry = _portal.DeckSolver._entry
    saved_basis = deck_solver.basis_entry

    def basis_entry(basis):
        cls, kwargs = saved_basis(basis)

        class Recording(cls):
            def __init__(self, **kw):  # no fill
                self.recorded = kw

        Recording.__name__ = cls.__name__
        return Recording, kwargs

    def entry(self, b):
        built.append(b)
        return {}

    deck_solver.basis_entry = basis_entry
    _portal.DeckSolver._entry = entry
    try:
        yield built
    finally:
        deck_solver.basis_entry = saved_basis
        _portal.DeckSolver._entry = saved_entry


_REFUSALS = (
    _portal.PortalError,
    ValueError,
    NotImplementedError,
    np.linalg.LinAlgError,
)


def _signs(b, model):
    """`_port_signs` over what the constructor was handed: the solver keeps
    `wires_polylines = [asarray(w) for w in wires]` and its `feeds` as given
    (only their types normalised), which is all `_port_signs` reads."""
    kw = b.solver.recorded
    shim = types.SimpleNamespace(
        solver=types.SimpleNamespace(
            wires_polylines=[np.asarray(w, dtype=float) for w in kw["wires"]],
            feeds=[(int(w), float(a), complex(v)) for w, a, v in kw["feeds"]],
        ),
        ports=b.ports,
    )
    return _portal._port_signs(shim, model)


def _dump_point(b, model, p, out):
    kw = dict(b.solver.recorded)
    out.append(f"{p}.kwargs : mapping {len(kw)}")
    for k, (name, value) in enumerate(kw.items()):
        dc.dump(name, f"{p}.kwargs{{{k}}}.key", out)
        dc.dump(_plain(value), f"{p}.kwargs{{{k}}}.value", out)
    dc.dump(b.ports, f"{p}.ports", out)
    dc.dump(b.site_to_solver_port, f"{p}.site_to_solver_port", out)
    dc.dump(b.frequency_mhz, f"{p}.frequency_mhz", out)
    dc.dump(b.wavelength, f"{p}.wavelength", out)
    dc.dump(b.group, f"{p}.group", out)
    dc.dump(b.extended_kernel, f"{p}.extended_kernel", out)
    dc.dump(b.advisories, f"{p}.advisories", out)
    dc.dump(_signs(b, model).tolist(), f"{p}.signs", out)


def _dump_tables(s, p, out):
    dc.dump(s.ports, f"{p}.ports", out)
    segments = [
        (
            g.number,
            g.tag,
            g.wire,
            g.local,
            g.centre.tolist(),
            g.direction.tolist(),
            g.radius,
        )
        for g in s.segments
    ]
    dc.dump(segments, f"{p}.segments", out)
    dc.dump(s._group, f"{p}.group", out)
    dc.dump(s._smallest_radius, f"{p}.smallest_radius", out)
    dc.dump(s.feed_index, f"{p}.feed_index", out)
    dc.dump(s.load_ports, f"{p}.load_ports", out)
    dc.dump(s._model_loads, f"{p}.model_loads", out)
    dc.dump(s._group_loads, f"{p}.group_loads", out)
    dc.dump(s.port_by_segment, f"{p}.port_by_segment", out)
    dc.dump(s._lossy, f"{p}.lossy", out)
    dc.dump(s._native_loading, f"{p}.native_loading", out)
    dc.dump(sorted(s._current_ports), f"{p}.current_ports", out)


def _build(seen: dict, basis: str, p: str, out: list[str]) -> bool:
    """One served structure's DeckSolver and its operating points; False
    when it refused or raised (the chain stops there)."""
    deck = seen["deck"]
    model = deck.model
    with _recording() as built, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            solver = _portal.DeckSolver(deck, basis=basis)
            for _index, group in seen["plan"]:
                for freq in group.freqs_mhz:
                    solver.at(
                        freq, extended_kernel=group.ek, environment=group.environment
                    )
        except _REFUSALS as exc:
            outcome = ("refused", exc)
        except Exception as exc:  # noqa: BLE001 — what escapes the slot
            outcome = ("raised", exc)
        else:
            outcome = None
    out.append(f"{p}.points : list {len(built)}")
    for j, b in enumerate(built):
        _dump_point(b, model, f"{p}.points[{j}]", out)
    if outcome is None:
        out.append(f"{p} = built")
        _dump_tables(solver, f"{p}.solver", out)
        return True
    kind, exc = outcome
    if kind == "refused":
        out.append(f"{p} = refused {dc._s(str(exc))}")
    else:
        out.append(dc._error(p, exc))
    return False


def _structure(text, echo, closing, basis, p, out) -> bool:
    with sc._stubbed() as seen:
        try:
            answer = _nec4.render_structure(
                text, basis=basis, echo_text=echo, closing=closing
            )
        except sc._Served:
            served = dict(seen)
        except Exception:  # noqa: BLE001 — the seam's own outcome (A2's gate)
            out.append(f"{p} = seam")
            return False
        else:
            del answer
            out.append(f"{p} = seam")
            return False
    return _build(served, basis, p, out)


def py_build(text: str, basis: str) -> str:
    out: list[str] = [f"build = basis {dc._s(basis)}"]
    try:
        structures = split_structures(text)
    except Exception:  # noqa: BLE001
        out.append("build = seam")
        return "".join(line + "\n" for line in out)
    items = (
        [(text, text, True)]
        if structures is None
        else [
            (s.deck_text, s.echo_text, k == len(structures) - 1 and not ends_on_nx(s))
            for k, s in enumerate(structures)
        ]
    )
    out.append(f"build.structures : list {len(items)}")
    for k, (t, e, closing) in enumerate(items):
        if not _structure(t, e, closing, basis, f"build.structures[{k}]", out):
            break
    return "".join(line + "\n" for line in out)


def cxx_build(text: str, basis: str, acc=None) -> str:
    return (acc or dc.accel()).nec4_build_dump(text.encode("latin-1"), basis)


def driver_build(driver: pathlib.Path, paths, basis: str) -> dict:
    run = subprocess.run(
        [str(driver), basis, *map(str, paths)], capture_output=True, check=True
    )
    dumps: dict[str, str] = {}
    current = None
    for line in run.stdout.decode("utf-8").splitlines(keepends=True):
        if line.startswith("== "):
            current = line[3:].rstrip("\n")
            dumps[current] = ""
        else:
            dumps[current] += line
    return dumps


def classify(dump: str) -> str:
    if " ? declined " in dump:
        return "declined"
    for line in dump.splitlines():
        if line == "build.structures[0] = built":
            return "built"
        if line.startswith("build.structures[0] = refused"):
            return "refused"
        if line.startswith("build.structures[0] ! "):
            return "raised"
        if line in ("build.structures[0] = seam", "build = seam"):
            return "seam"
    return "other"


def deck_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in SUFFIXES]


def compare(paths, basis: str, driver: pathlib.Path | None = None):
    counts = {
        "built": 0,
        "refused": 0,
        "raised": 0,
        "seam": 0,
        "declined": 0,
        "mismatch": 0,
    }
    mismatches: list[tuple[str, str]] = []
    texts = {str(p): p.read_bytes().decode("latin-1") for p in paths}
    if driver is not None:
        cxx = driver_build(driver, list(paths), basis)
    else:
        acc = dc.accel()
        cxx = {k: cxx_build(t, basis, acc) for k, t in texts.items()}
    for label, text in texts.items():
        c = cxx[label]
        kind = classify(c)
        if kind == "declined":
            counts["declined"] += 1
            continue
        diff = dc.first_difference(py_build(text, basis), c)
        if diff is None:
            counts[kind if kind in counts else "built"] += 1
        else:
            counts["mismatch"] += 1
            mismatches.append((label, diff))
    return counts, mismatches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--basis", choices=(*BASES, "both"), default="both")
    ap.add_argument("--driver", type=pathlib.Path)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args(argv)
    bases = BASES if args.basis == "both" else (args.basis,)
    bad = 0
    for root in args.dirs:
        paths = deck_files(root)
        for basis in bases:
            counts, mismatches = compare(paths, basis, args.driver)
            print(f"{root} [{basis}]: {counts}", flush=True)
            for label, diff in mismatches[: args.show]:
                print(f"  MISMATCH {label}\n{diff}", flush=True)
            bad += counts["mismatch"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
