"""The C++ NEC-5 deck reader against the Python one, deck by deck.

`src/momwire/_nec5_deck_core.cpp` is a copy of momwire's NEC-5 reader
(`deck/_cards.py`, `deck/_nec5.py`) and of the EZNEC seam's text-level front
(`_serve.split_runs`, `_shell._answer` / `render` up to the solve). This
module writes the Python side of the comparison format the core's
`dump_parse` / `dump_stream` write -- one line per field, floats as their 64
bits, ints in decimal, strings as UTF-8 hex, an exception as its type and
message -- so two readers agree exactly when their dumps are equal, and the
first differing line names the field.

The Python side is generic (`dump`): it walks every dataclass field the
reference returns, so a field added to `Nec5Deck` shows up as a mismatch
until the C++ carries it too. `py_stream` drives the seam's REAL control flow
(`_shell._answer`, `split_structures`, `split_runs`, `parse_nec5_shared`)
with the solve and the refusal printout stubbed out, so the order in which it
raises, refuses and serves is the seam's own rather than a re-statement.

    python scripts/core_split/deck_compare.py DIR [DIR ...] [--show 3]

reads every `*.nec` / `*.deck` file under each DIR (latin-1, as the seam
reads a deck), and reports per directory: parsed identically, refused
identically (a DeckError, by message), raised identically (any other
exception, by type and message), and mismatches. Exit status 1 on any
mismatch. `--texts FILE.jsonl` reads decks as JSON strings instead, one per
line (the harvest written by `nec5_harvest_plugin.py`).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import pathlib
import struct
import sys
from collections.abc import Mapping

from momwire.deck._nec5 import parse_nec5, split_structures
from momwire.eznec import _printout, _serve, _shell

SUFFIXES = {".nec", ".deck"}


def _f(x: float) -> str:
    return struct.pack(">d", x).hex()


def _s(x: str) -> str:
    return x.encode("utf-8").hex()


def dump(obj, path: str, out: list[str]) -> None:
    """``obj`` in the comparison format, one line per field."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out.append(f"{path} : {type(obj).__name__}")
        for field in dataclasses.fields(obj):
            dump(getattr(obj, field.name), f"{path}.{field.name}", out)
    elif obj is None:
        out.append(f"{path} = None")
    elif isinstance(obj, bool):
        out.append(f"{path} = bool {obj}")
    elif isinstance(obj, int):
        out.append(f"{path} = int {obj}")
    elif isinstance(obj, float):
        out.append(f"{path} = float {_f(obj)}")
    elif isinstance(obj, complex):
        out.append(f"{path} = complex {_f(obj.real)} {_f(obj.imag)}")
    elif isinstance(obj, str):
        out.append(f"{path} = str {_s(obj)}")
    elif isinstance(obj, (tuple, list)):
        kind = "tuple" if isinstance(obj, tuple) else "list"
        out.append(f"{path} : {kind} {len(obj)}")
        for k, item in enumerate(obj):
            dump(item, f"{path}[{k}]", out)
    elif isinstance(obj, Mapping):
        out.append(f"{path} : mapping {len(obj)}")
        for k, (key, value) in enumerate(obj.items()):
            dump(key, f"{path}{{{k}}}.key", out)
            dump(value, f"{path}{{{k}}}.value", out)
    else:
        raise TypeError(f"{path}: no comparison spelling for {type(obj).__name__}")


def _error(path: str, exc: BaseException) -> str:
    return f"{path} ! {type(exc).__name__} {_s(str(exc))}"


def py_parse(text: str) -> str:
    """`parse_nec5(text)`'s outcome in the comparison format."""
    out: list[str] = []
    try:
        deck = parse_nec5(text)
    except Exception as exc:  # noqa: BLE001 — the outcome IS the exception
        out.append(_error("parse", exc))
    else:
        dump(deck, "parse", out)
    return "".join(line + "\n" for line in out)


class _Refused(str):
    """What the stubbed refusal printout returns: the refusal's message."""

    __slots__ = ()


class _Served:
    def __init__(self, decks):
        self.decks = decks


@contextlib.contextmanager
def _stubbed_seam():
    """`_shell._answer` with the solve and the refusal printout replaced:
    `serve` / `serve_runs` record the decks they were handed, and
    `render_refusal` returns its message."""
    saved = (_serve.serve, _serve.serve_runs, _printout.render_refusal)
    _serve.serve = lambda deck, **_: _Served([deck])
    _serve.serve_runs = lambda decks, **_: [_Served(list(decks))]
    _printout.render_refusal = lambda echo, message, **_: _Refused(message)
    try:
        yield
    finally:
        _serve.serve, _serve.serve_runs, _printout.render_refusal = saved


def _answer_lines(text: str, echo: str | None, path: str, out: list[str]) -> bool:
    """One structure through `_shell._answer`; False when it refused or raised."""
    try:
        answered = _shell._answer(text, basis=_serve.BASIS, echo_text=echo)
    except Exception as exc:  # noqa: BLE001 — what escapes the seam's front
        out.append(_error(path, exc))
        return False
    if isinstance(answered, _Refused):
        out.append(f"{path} = refused {_s(str(answered))}")
        return False
    deck, answers, layout = answered
    if layout == "stepped":
        # momwire#1508: the one run at each frequency (the deck carries them)
        out.append(f"{path} = stepped")
        dump(deck, f"{path}.deck", out)
    elif layout == "multi":
        out.append(f"{path} = multi")
        dump(deck, f"{path}.deck", out)
        (served,) = answers
        dump(list(served.decks), f"{path}.runs", out)
    else:
        out.append(f"{path} = single")
        dump(deck, f"{path}.deck", out)
    return True


def py_stream(text: str) -> str:
    """`_shell.render(text)`'s front, up to the solve, in the comparison
    format: the NX structures and each one's answer, stopping where
    `_render_chain` stops."""
    out: list[str] = []
    structures = split_structures(text)
    answers: list[str] = []
    count = 0
    with _stubbed_seam():
        if structures is None:
            out.append("stream.structures = None")
            count = 1
            _answer_lines(text, None, "stream.answers[0]", answers)
        else:
            out.append(f"stream.structures : list {len(structures)}")
            for k, s in enumerate(structures):
                out.append(f"stream.structures[{k}].body = str {_s(s.body)}")
                term = s.terminator
                out.append(
                    f"stream.structures[{k}].terminator = "
                    + ("None" if term is None else f"str {_s(term)}")
                )
            for k, s in enumerate(structures):
                count += 1
                path = f"stream.answers[{k}]"
                if not _answer_lines(s.deck_text, s.echo_text, path, answers):
                    break
    out.append(f"stream.answers : list {count}")
    out.extend(answers)
    return "".join(line + "\n" for line in out)


def accel():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec5_deck_dump"):
        raise SystemExit("no compiled accelerator with the NEC-5 deck reader")
    return _accel.acc


def cxx_parse(text: str, acc=None) -> str:
    return (acc or accel()).nec5_deck_dump(text.encode("latin-1"), "parse")


def cxx_stream(text: str, acc=None) -> str:
    return (acc or accel()).nec5_deck_dump(text.encode("latin-1"), "stream")


def first_difference(a: str, b: str) -> str | None:
    """The first differing line of two dumps, both sides, or None."""
    la, lb = a.splitlines(), b.splitlines()
    for k in range(max(len(la), len(lb))):
        x = la[k] if k < len(la) else "<missing>"
        y = lb[k] if k < len(lb) else "<missing>"
        if x != y:
            return f"line {k}: python {x!r}\n          c++    {y!r}"
    return None


def classify(text: str, acc=None) -> tuple[str, str | None]:
    """``(category, difference)`` for one deck: category is ``parsed``,
    ``refused``, ``raised`` or ``mismatch``, by the Python parse."""
    ref_parse, ref_stream = py_parse(text), py_stream(text)
    diff = first_difference(ref_parse, cxx_parse(text, acc))
    if diff is None:
        diff = first_difference(ref_stream, cxx_stream(text, acc))
    if diff is not None:
        return "mismatch", diff
    head = ref_parse.split("\n", 1)[0]
    if head.startswith("parse ! DeckError"):
        return "refused", None
    if head.startswith("parse !"):
        return "raised", None
    return "parsed", None


def deck_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="*", type=pathlib.Path)
    ap.add_argument("--texts", type=pathlib.Path, help="a JSONL of deck texts")
    ap.add_argument("--show", type=int, default=5, help="mismatches to print")
    args = ap.parse_args(argv)
    acc = accel()
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    for root in args.dirs:
        items = [(str(p), p.read_bytes().decode("latin-1")) for p in deck_files(root)]
        groups.append((str(root), items))
    if args.texts:
        lines = args.texts.read_text().splitlines()
        items = [
            (f"{args.texts}:{k}", json.loads(line)) for k, line in enumerate(lines)
        ]
        groups.append((str(args.texts), items))
    total = {"parsed": 0, "refused": 0, "raised": 0, "mismatch": 0, "skipped": 0}
    shown = 0
    for name, items in groups:
        counts = dict.fromkeys(total, 0)
        for label, text in items:
            try:
                text.encode("latin-1")
            except UnicodeEncodeError:
                counts["skipped"] += 1  # not a deck the seam can read
                continue
            category, diff = classify(text, acc)
            counts[category] += 1
            if diff is not None and shown < args.show:
                shown += 1
                print(f"MISMATCH {label}\n  {diff}")
        for k in total:
            total[k] += counts[k]
        print(
            f"{name}: {len(items)} decks: parsed identically {counts['parsed']}, "
            f"refused identically {counts['refused']}, raised identically "
            f"{counts['raised']}, mismatches {counts['mismatch']}"
            + (f", not latin-1 {counts['skipped']}" if counts["skipped"] else "")
        )
    print(
        f"TOTAL: parsed identically {total['parsed']}, refused identically "
        f"{total['refused']}, raised identically {total['raised']}, "
        f"mismatches {total['mismatch']}"
        + (f", not latin-1 {total['skipped']}" if total["skipped"] else "")
    )
    return 1 if total["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
