"""The C++ nec2 / nec4 deck reader against the Python one, deck by deck.

`src/momwire/_nec4_deck_core.cpp` is a copy of momwire's reader for the
segment-centre dialects (`deck/_nec2.py`, `deck/_nec2_geometry.py` and the
NEC-4.2 slot's deltas in `deck/_nec4.py`), the first unit of the standalone
NEC-4.2-dialect executables (momwire#1607 A1). This module writes the Python
side of the comparison format the core's `dump_parse` writes --
`deck_compare.py`'s generic dataclass walk over `momwire.deck.parse(text,
dialect=...)`'s DeckModel, one line per field, floats as their 64 bits -- so
the two readers agree exactly when their dumps are equal, and the first
differing line names the field. A field added to `DeckModel` shows up as a
mismatch until the C++ carries it too.

    python scripts/core_split/nec4_deck_compare.py DIR [DIR ...] \\
        [--dialect nec4|nec2|both] [--driver PATH] [--show 5]

reads every `*.nec` / `*.deck` file under each DIR (latin-1) and reports per
directory and dialect: parsed identically, refused identically (by exception
type and message), declined by the C++ (counted, never compared) and
mismatches. Exit status 1 on any mismatch. The C++ side is the extension's
`nec4_deck_dump`, or with `--driver` the Python-free
`nec4_deck_driver` built from the core TUs alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys
import tempfile

from momwire.deck import parse

HERE = pathlib.Path(__file__).resolve().parent
SUFFIXES = {".nec", ".deck"}
DIALECTS = ("nec2", "nec4")


def _deck_compare():
    spec = importlib.util.spec_from_file_location(
        "core_split_deck_compare", HERE / "deck_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dc = _deck_compare()


def py_parse(text: str, dialect: str) -> str:
    """`momwire.deck.parse(text, dialect)`'s outcome in the comparison format."""
    out: list[str] = []
    try:
        model = parse(text, dialect=dialect)
    except Exception as exc:  # noqa: BLE001 — the outcome IS the exception
        out.append(dc._error("parse", exc))
    else:
        dc.dump(model, "parse", out)
    return "".join(line + "\n" for line in out)


def cxx_parse(text: str, dialect: str, acc=None) -> str:
    """The extension's C++ reader on the same text."""
    return (acc or dc.accel()).nec4_deck_dump(text.encode("latin-1"), dialect)


def driver_parse(driver: pathlib.Path, paths: list[pathlib.Path], dialect: str) -> dict:
    """The Python-free driver's dumps, one per deck file, keyed by path."""
    run = subprocess.run(
        [str(driver), dialect, *map(str, paths)], capture_output=True, check=True
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
    first = dump.split("\n", 1)[0]
    if first.startswith("parse ? declined"):
        return "declined"
    if first.startswith("parse ! "):
        return "refused"
    return "parsed"


def deck_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in SUFFIXES]


def compare(
    paths: list[pathlib.Path], dialect: str, driver: pathlib.Path | None = None
) -> tuple[dict, list[tuple[str, str]]]:
    """Counts per outcome and the mismatches (label, first difference)."""
    counts = {"parsed": 0, "refused": 0, "declined": 0, "mismatch": 0}
    mismatches: list[tuple[str, str]] = []
    texts = {str(p): p.read_bytes().decode("latin-1") for p in paths}
    if driver is not None:
        with tempfile.TemporaryDirectory():
            cxx = driver_parse(driver, paths, dialect)
    else:
        acc = dc.accel()
        cxx = {k: cxx_parse(t, dialect, acc) for k, t in texts.items()}
    for label, text in texts.items():
        c = cxx[label]
        kind = classify(c)
        if kind == "declined":
            counts["declined"] += 1
            continue
        diff = dc.first_difference(py_parse(text, dialect), c)
        if diff is None:
            counts[kind] += 1
        else:
            counts["mismatch"] += 1
            mismatches.append((label, diff))
    return counts, mismatches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--dialect", choices=(*DIALECTS, "both"), default="both")
    ap.add_argument("--driver", type=pathlib.Path)
    ap.add_argument("--show", type=int, default=5, help="mismatches to print")
    args = ap.parse_args(argv)
    dialects = DIALECTS if args.dialect == "both" else (args.dialect,)
    bad = 0
    for root in args.dirs:
        paths = deck_files(root)
        for dialect in dialects:
            counts, mismatches = compare(paths, dialect, args.driver)
            print(f"{root} [{dialect}]: {counts}")
            for label, diff in mismatches[: args.show]:
                print(f"  MISMATCH {label}\n{diff}")
            bad += counts["mismatch"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
