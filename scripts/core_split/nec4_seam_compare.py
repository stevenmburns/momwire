"""The C++ NEC-4.2 seam front against the Python one, deck by deck.

`src/momwire/_nec4_seam_core.cpp` copies the text-level front of the
NEC-4.2 slot (momwire#1607 A2), everything `eznec/_nec4.py`'s `render` does
before the solve:
- the basis gate;
- the NX chain split;
- `portal._portal.parse_deck`'s PortalDeck, the printout's view of the
  cards;
- `_further_tables`, the echo numbers of the runs and the tables asked of a
  run after it ran;
- `_served_shape`, the runs to solve or the refusal naming a shape no
  NEC-4.2 printout shows.

This module writes the Python side of the comparison format the core's
`dump_seam` writes. `py_seam` drives the slot's REAL control flow
(`_nec4.render_structure` and `solve`) with the solve and the refusal
printout stubbed out: `DeckSolver` stops the run where the build would start
(after `_served_shape` returned), and `render_refusal` hands back its
message. So the order in which a structure refuses, raises or is served is
the slot's own, not a restatement. Only `_render_chain`'s loop is
restated, so that a served structure lets the next one run.

    python scripts/core_split/nec4_seam_compare.py DIR [DIR ...] \\
        [--basis sinusoidal|bspline] [--driver PATH] [--show 5]

Exit status 1 on any mismatch. The C++ side is the extension's
`nec4_seam_dump`, or with `--driver` the Python-free `nec4_seam_driver`.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import pathlib
import subprocess
import sys

from momwire.deck._nec5 import ends_on_nx, split_structures
from momwire.eznec import _nec4, _printout
from momwire.portal import _portal

HERE = pathlib.Path(__file__).resolve().parent
SUFFIXES = {".nec", ".deck"}
BASES = ("sinusoidal", "bspline")


def _deck_compare():
    spec = importlib.util.spec_from_file_location(
        "core_split_deck_compare", HERE / "deck_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dc = _deck_compare()


class _Refused(str):
    __slots__ = ()


class _Served(BaseException):  # noqa: N818 — a control-flow signal, not an error
    """Raised where the build would start: the structure was served."""


@contextlib.contextmanager
def _stubbed():
    """`_nec4.solve` stopped at `DeckSolver`, with what it read recorded, and
    `render_refusal` returning its message."""
    seen: dict = {}
    saved = (
        _portal.DeckSolver,
        _portal.parse_deck,
        _nec4._further_tables,
        _nec4._served_shape,
        _printout.render_refusal,
    )
    real_parse, real_further, real_shape = saved[1], saved[2], saved[3]

    def parse_deck(text, dialect="nec2"):
        seen["deck"] = real_parse(text, dialect=dialect)
        return seen["deck"]

    def further(text, deck):
        seen["further"] = real_further(text, deck)
        return seen["further"]

    def shape(text, deck, echo_at, further_):
        seen["plan"] = real_shape(text, deck, echo_at, further_)
        return seen["plan"]

    def solver(deck, basis):
        raise _Served

    _portal.DeckSolver = solver
    _portal.parse_deck = parse_deck
    _nec4._further_tables = further
    _nec4._served_shape = shape
    _printout.render_refusal = lambda echo, message, **_: _Refused(message)
    try:
        yield seen
    finally:
        (
            _portal.DeckSolver,
            _portal.parse_deck,
            _nec4._further_tables,
            _nec4._served_shape,
            _printout.render_refusal,
        ) = saved


def _dump_served(seen: dict, p: str, out: list[str]) -> None:
    deck = seen["deck"]
    out.append(f"{p} = served")
    view = dataclasses.replace(deck, model=None, structure=None)
    dc.dump(view, f"{p}.deck", out)
    st = deck.structure
    q = f"{p}.structure"
    dc.dump(st.wires, f"{q}.wires", out)
    dc.dump(st.pieces, f"{q}.pieces", out)
    dc.dump(st.pieces_of_wire, f"{q}.pieces_of_wire", out)
    dc.dump(st.ground_plane_flag, f"{q}.ground_plane_flag", out)
    dc.dump(st.ground_plane_interpolates, f"{q}.ground_plane_interpolates", out)
    dc.dump(st.symmetry, f"{q}.symmetry", out)
    echo_at, further = seen["further"]
    dc.dump(echo_at, f"{p}.echo_at", out)
    dc.dump(further, f"{p}.further", out)
    dc.dump([index for index, _group in seen["plan"]], f"{p}.plan", out)


def _structure(text, echo, closing, basis, p, out) -> bool:
    """One structure through `render_structure`; False when the chain stops."""
    dc.dump(text, f"{p}.deck_text", out)
    dc.dump(echo, f"{p}.echo_text", out)
    dc.dump(closing, f"{p}.closing", out)
    with _stubbed() as seen:
        try:
            answer = _nec4.render_structure(
                text, basis=basis, echo_text=echo, closing=closing
            )
        except _Served:
            _dump_served(seen, p, out)
            return True
        except Exception as exc:  # noqa: BLE001 — what escapes the slot's front
            out.append(dc._error(p, exc))
            return False
    printout, _seconds = answer
    out.append(f"{p} = refused {dc._s(str(printout))}")
    return False


def py_seam(text: str, basis: str) -> str:
    """`_nec4.render(text, basis=...)` up to the solve, in the comparison format."""
    out: list[str] = [f"seam = basis {dc._s(basis)}"]
    refused = _nec4.basis_refusal(basis)
    if refused is not None:
        out.append(f"seam = refused {dc._s(refused)}")
        return "".join(line + "\n" for line in out)
    try:
        structures = split_structures(text)
    except Exception as exc:  # noqa: BLE001
        out.append(dc._error("seam", exc))
        return "".join(line + "\n" for line in out)
    if structures is None:
        out.append("seam.chained = bool False")
        out.append("seam.structures : list 1")
        _structure(text, text, True, basis, "seam.structures[0]", out)
    else:
        out.append("seam.chained = bool True")
        out.append(f"seam.structures : list {len(structures)}")
        for k, s in enumerate(structures):
            last = k == len(structures) - 1 and not ends_on_nx(s)
            if not _structure(
                s.deck_text, s.echo_text, last, basis, f"seam.structures[{k}]", out
            ):
                break
    return "".join(line + "\n" for line in out)


def cxx_seam(text: str, basis: str, acc=None) -> str:
    return (acc or dc.accel()).nec4_seam_dump(text.encode("latin-1"), basis)


def driver_seam(driver: pathlib.Path, paths: list[pathlib.Path], basis: str) -> dict:
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
    """served / refused / raised / declined, by the first structure's line
    (the deck's outcome where it is one structure)."""
    if " ? declined " in dump:
        return "declined"
    for line in dump.splitlines():
        if line.startswith("seam.structures[0] = served"):
            return "served"
        if line.startswith("seam.structures[0] = refused") or line.startswith(
            "seam = refused"
        ):
            return "refused"
        if line.startswith("seam.structures[0] ! ") or line.startswith("seam ! "):
            return "raised"
    return "other"


def deck_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in SUFFIXES]


def compare(paths, basis: str, driver: pathlib.Path | None = None):
    counts = {"served": 0, "refused": 0, "raised": 0, "declined": 0, "mismatch": 0}
    mismatches: list[tuple[str, str]] = []
    texts = {str(p): p.read_bytes().decode("latin-1") for p in paths}
    if driver is not None:
        cxx = driver_seam(driver, list(paths), basis)
    else:
        acc = dc.accel()
        cxx = {k: cxx_seam(t, basis, acc) for k, t in texts.items()}
    for label, text in texts.items():
        c = cxx[label]
        kind = classify(c)
        if kind == "declined":
            counts["declined"] += 1
            continue
        diff = dc.first_difference(py_seam(text, basis), c)
        if diff is None:
            counts[kind if kind in counts else "served"] += 1
        else:
            counts["mismatch"] += 1
            mismatches.append((label, diff))
    return counts, mismatches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--basis", choices=(*BASES, "both"), default="sinusoidal")
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
