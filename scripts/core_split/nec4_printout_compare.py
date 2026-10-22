"""The C++ NEC-4.2 printout writer against the Python one, deck by deck.

`src/momwire/_nec4_printout_core.cpp` copies `eznec/_nec4_printout.py`'s
`render_nec4_printout` and the shared `_printout` sections it uses
(momwire#1607 A3). It is a FORMATTER. The gate therefore hands it the very
`Nec4Printout` Python's own solve built:
- `_nec4.render` runs for real;
- `render_nec4_printout` is wrapped to record each call's
  `(echo, data, basis, closing)` and Python's text;
- each recorded call is replayed through the extension's
  `nec4_render_printout` and the two texts are compared.

A deck the slot refuses before any solve is compared whole, with no Python
in the C++ path. The C++ seam and build (A2, A2b) decide the refusal, and the
C++ writes the refusal printout; `_nec4.render`'s text is the reference.

    python scripts/core_split/nec4_printout_compare.py DIR [DIR ...] \\
        [--basis sinusoidal|bspline|both] [--refusals-only] [--show 3] [--out FILE]

Counts per directory and basis: printouts compared and equal, refusal
printouts compared and equal, mismatches, and decks neither path printed
(raised, declined). Exit status 1 on any mismatch. Results are written one
line per deck, flushed, when `--out` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import sys
import warnings

from momwire.eznec import _nec4, _printout

SUFFIXES = {".nec", ".deck"}
BASES = ("sinusoidal", "bspline")


def accel():
    from momwire import _accel

    return _accel.acc


def stamp() -> tuple[str, str, str]:
    """The three fields Python's line-2 stamp is built from."""
    version = _printout._stamp_field(_printout._momwire_version()) or "unknown"
    variant = _printout._stamp_field(_printout._accelerator_variant()) or "none"
    return version, variant, _printout._overlay_marker(version)


@contextlib.contextmanager
def recording():
    calls: list[tuple[str, object, str, bool, str]] = []
    real = _nec4.render_nec4_printout

    def render(deck_text, data, *, basis=None, closing=True):
        text = real(deck_text, data, basis=basis, closing=closing)
        calls.append((deck_text, data, basis, closing, text))
        return text

    _nec4.render_nec4_printout = render
    try:
        yield calls
    finally:
        _nec4.render_nec4_printout = real


def first_difference(a: str, b: str) -> str | None:
    la, lb = a.split("\n"), b.split("\n")
    for k in range(max(len(la), len(lb))):
        x = la[k] if k < len(la) else "<missing>"
        y = lb[k] if k < len(lb) else "<missing>"
        if x != y:
            return f"line {k}: python {x!r}\n          c++    {y!r}"
    return None


def compare_deck(
    text: str, basis: str, acc, refusals_only: bool = False
) -> tuple[str, str | None]:
    """``(outcome, difference)``: outcome is ``refusal`` / ``printout`` (equal),
    ``mismatch`` (with the first difference), or ``skipped``."""
    version, variant, marker = stamp()
    blob = text.encode("latin-1")
    refused = acc.nec4_refusal_printout(blob, basis, version, variant, marker)
    if refused is not None:
        python = _nec4.render(text, basis=basis)
        diff = first_difference(python, refused)
        return ("refusal", None) if diff is None else ("mismatch", diff)
    if refusals_only:
        return "skipped", None
    if " ? declined " in acc.nec4_build_dump(blob, basis):
        # The C++ declines it (a non-finite field, sizes past what it
        # materialises): nothing to compare, and Python's solve of such a
        # deck can run for a long time or exhaust memory.
        return "skipped", None
    with recording() as calls, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _nec4.render(text, basis=basis)
        except Exception:  # noqa: BLE001 — a raised deck has no printout to compare
            return "skipped", None
    if not calls:
        return "skipped", None
    for echo, data, b, closing, python in calls:
        cxx = acc.nec4_render_printout(
            echo.encode("latin-1"), data, b, version, variant, marker, closing
        )
        diff = first_difference(python, cxx)
        if diff is not None:
            return "mismatch", diff
    return "printout", None


def deck_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in SUFFIXES]


def compare(paths, basis: str, refusals_only: bool = False, out=None):
    acc = accel()
    counts = {"printout": 0, "refusal": 0, "mismatch": 0, "skipped": 0}
    mismatches: list[tuple[str, str]] = []
    for path in paths:
        text = path.read_bytes().decode("latin-1")
        kind, diff = compare_deck(text, basis, acc, refusals_only)
        counts[kind] += 1
        if diff is not None:
            mismatches.append((str(path), diff))
        if out is not None:
            print(
                basis,
                kind,
                path,
                "" if diff is None else diff.replace("\n", " | "),
                file=out,
                flush=True,
            )
    return counts, mismatches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--basis", choices=(*BASES, "both"), default="sinusoidal")
    ap.add_argument("--refusals-only", action="store_true")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args(argv)
    bases = BASES if args.basis == "both" else (args.basis,)
    out = open(args.out, "w") if args.out else None  # noqa: SIM115 — closed below
    bad = 0
    try:
        for root in args.dirs:
            paths = deck_files(root)
            for basis in bases:
                counts, mismatches = compare(paths, basis, args.refusals_only, out)
                print(f"{root} [{basis}]: {counts}", flush=True)
                for label, diff in mismatches[: args.show]:
                    print(f"  MISMATCH {label}\n{diff}", flush=True)
                bad += counts["mismatch"]
    finally:
        if out is not None:
            out.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
