#!/usr/bin/env python3
"""Write ``support/FEATURES.md``: every experimental feature and how far it
is promoted, generated from the code.

Sources, all read, none copied:

- ``momwire._features.FEATURES``: issue, since, kind, reach, default, offer,
  declared drift;
- ``support/hierarchy.toml``: the level ids and descriptions;
- ``standalone/momwire_razor2p.cpp``: the ``EXE_FEATURES`` rows (what the
  executables carry) and the ``FEATURE_ALIASES`` rows (older spellings the
  executables know and refuse);
- ``CXX`` below: whether C++ exists for a feature. No registry field says so
  (``kind`` is close but is not it: ``height_sheet_recal`` is a Python-kind
  route the executables' C++ carries), so it is a small explicit table here.
  ``tests/test_feature_table.py`` holds its "exe route" rows equal to the
  fields of ``struct Features`` in ``src/momwire/_near_interface_eval.h``
  and every file it names to an existing one.

Run from the repo root, in a venv with this checkout installed::

    python scripts/feature_table.py          # rewrite support/FEATURES.md
    python scripts/feature_table.py --check  # exit 1 when it is stale

``tests/test_feature_table.py`` regenerates it and fails when it differs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIERARCHY = ROOT / "support" / "hierarchy.toml"
EXE_SOURCE = ROOT / "standalone" / "momwire_razor2p.cpp"
OUTPUT = ROOT / "support" / "FEATURES.md"
REGENERATE = "python scripts/feature_table.py"

# Whether C++ exists for each feature, and where:
#   "exe"       -- a switch of the near-interface evaluator's C++ route
#                  (`ni_eval::Features`), the one the executables build, so a
#                  promotion past level 5 is a table row and its gates, not a
#                  port;
#   "extension" -- C++ in the Python extension only (the files named); the
#                  executables' routes do not carry it;
#   "none"      -- a Python route over shipped kernels; no C++ of its own.
# Edit this when a feature gains or loses C++; the test fails on a feature
# missing here, on an "exe" row the evaluator does not declare, and on a
# named file that does not exist.
CXX: dict[str, tuple[str, tuple[str, ...]]] = {
    "pair_reuse": (
        "extension",
        ("src/momwire/_accel_somm.cpp", "src/momwire/_accel_sin_pair_reuse.cpp"),
    ),
    "offset_reuse": ("none", ()),
    "transmitted_anchors": ("none", ()),
    "point_cplx_staged": (
        "extension",
        (
            "src/momwire/_accel_sinusoidal.cpp",
            "src/momwire/_sinusoidal_galerkin_core.cpp",
        ),
    ),
    "factored_columns": ("exe", ("src/momwire/_near_interface_eval.cpp",)),
    "height_sheet_recal": ("exe", ("src/momwire/_near_interface_eval.cpp",)),
    "crossing_table": ("exe", ("src/momwire/_near_interface_eval.cpp",)),
    "crossing_above_rule": ("exe", ("src/momwire/_near_interface_eval.cpp",)),
    "crossing_sheet_retune": ("exe", ("src/momwire/_near_interface_eval.cpp",)),
    "kcl_nullspace": ("none", ()),
    "exact_kernel_series": (
        "extension",
        ("src/momwire/_accel_exact_kernel_series.cpp",),
    ),
    "razor_cxx": (
        "extension",
        ("src/momwire/_razor_solve_core.cpp", "src/momwire/_razor_front_core.cpp"),
    ),
}


def load_levels(path: Path = HIERARCHY) -> list[tuple[str, str]]:
    """``hierarchy.toml``'s levels, in order: [(id, description)]."""
    try:
        import tomllib  # Python 3.11+; 3.10 falls back to tomli
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [(lv["id"], lv["description"]) for lv in data["level"]]


def exe_rows(src: str) -> dict[str, tuple[int, bool]]:
    """``EXE_FEATURES``: {name: (reach, default_on)}."""
    table = re.search(r"const ExeFeature EXE_FEATURES\[\] = \{(.*?)\n\};", src, re.S)
    if table is None:
        raise SystemExit(f"EXE_FEATURES not found in {EXE_SOURCE}")
    rows = re.findall(r'\{"([a-z0-9_]+)", (\d+), (true|false)\}', table[1])
    if len(rows) != table[1].count("{"):
        raise SystemExit(f"an EXE_FEATURES row did not parse: {table[1]!r}")
    return {n: (int(r), d == "true") for n, r, d in rows}


def exe_aliases(src: str) -> dict[str, list[str]]:
    """``FEATURE_ALIASES``: {feature name: [variable, ...]}."""
    table = re.search(r"FEATURE_ALIASES\[\]\[2\] = \{(.*?)\n\};", src, re.S)
    if table is None:
        raise SystemExit(f"FEATURE_ALIASES not found in {EXE_SOURCE}")
    out: dict[str, list[str]] = {}
    for var, name in re.findall(
        r'\{"(MOMWIRE_[A-Z0-9_]+)", "([a-z0-9_]+)"\}', table[1]
    ):
        out.setdefault(name, []).append(var)
    return out


def _level(levels: list[tuple[str, str]], number: int) -> str:
    return f"{number} `{levels[number - 1][0]}`"


def _span(numbers: list[int]) -> str:
    if not numbers:
        return "none"
    if len(numbers) == 1:
        return str(numbers[0])
    return f"{numbers[0]}-{numbers[-1]}"


def _exe_cell(
    name: str,
    rows: dict[str, tuple[int, bool]],
    aliases: dict[str, list[str]],
    exe_levels: list[int],
) -> str:
    if name in rows:
        reach, _default = rows[name]
        runs = [n for n in exe_levels if n <= reach]
        refused = [n for n in exe_levels if n > reach]
        cell = f"`EXE_FEATURES` row: runs at {_span(runs)}"
        if refused:
            cell += f", refused at {_span(refused)}"
        return cell
    if name in aliases:
        spelled = ", ".join(f"`{v}`" for v in aliases[name])
        return f"not carried; alias {spelled} known and refused"
    return "not carried; refused"


def _drift(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _cxx_cell(name: str) -> str:
    where, files = CXX[name]
    if where == "none":
        return "none"
    label = "exe route" if where == "exe" else "extension only"
    names = ", ".join(f"`{Path(f).name}`" for f in files)
    return f"{label} ({names})"


HEADER = """\
# Experimental features

Generated by `{regenerate}` from `src/momwire/_features.py`,
`support/hierarchy.toml`, `standalone/momwire_razor2p.cpp` and the script's
own `CXX` table. Do not edit by hand: `tests/test_feature_table.py` fails
when this file is stale.

## Levels

A feature of reach N is certified on levels 1..N. **Stable** means reach ==
the deepest level, {deepest}.

| level | id | description |
|---|---|---|
{level_rows}

**Promotion** is raising a feature's `reach`, one level at a time, in its own
PR that selects the new levels' jobs; it is Steve's call. **Default-on** is a
separate promotion (`default`), also Steve's, and independent of reach.

## Columns

- **kind**: outside its reach, `python` runs and is stamped uncertified on
  printout line 2; `native` refuses with `FeatureNotCertified`.
- **drift**: the declared bound on |dZin|/|Zin|, on against off.
- **offer**: whether a front end exposes the switch per request.
- **executables**: the levels (of {exe_span}) where the standalone executables
  run it, from `EXE_FEATURES`; any other feature turned on makes them exit 2.
- **C++**: `exe route` is a switch of the C++ route the executables build;
  `extension only` is C++ in the Python extension alone; `none` is Python
  only.

## Features
"""


def render(
    features: dict,
    levels: list[tuple[str, str]],
    exe_source: str,
    deepest: int,
) -> str:
    """The whole of ``support/FEATURES.md``."""
    rows = exe_rows(exe_source)
    aliases = exe_aliases(exe_source)
    exe_levels = [
        n for n, (lid, _d) in enumerate(levels, start=1) if lid.startswith("exe-")
    ]
    missing = sorted(set(features) - set(CXX))
    if missing:
        raise SystemExit(f"add {missing} to CXX in scripts/feature_table.py")
    level_rows = "\n".join(
        f"| {n}{' (stable)' if n == deepest else ''} | `{lid}` | {desc} |"
        for n, (lid, desc) in enumerate(levels, start=1)
    )
    out = [
        HEADER.format(
            regenerate=REGENERATE,
            deepest=deepest,
            level_rows=level_rows,
            exe_span=_span(exe_levels),
        ),
        "| feature | issue | since | kind | reach | default | offer | drift"
        " | executables | C++ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in features.values():
        reach = _level(levels, f.reach) + (" (stable)" if f.reach == deepest else "")
        out.append(
            f"| `{f.name}` | #{f.issue} | {f.since} | {f.kind} | {reach}"
            f" | {'on' if f.default else 'off'} | {'yes' if f.offer else 'no'}"
            f" | {_drift(f.drift)} | {_exe_cell(f.name, rows, aliases, exe_levels)}"
            f" | {_cxx_cell(f.name)} |"
        )
    return "\n".join(out) + "\n"


def generate() -> str:
    # Imported here so --help and the test's render() need no momwire import.
    from momwire import _features

    return render(
        _features.FEATURES,
        load_levels(),
        EXE_SOURCE.read_text(encoding="utf-8"),
        _features.DEEPEST,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    args = parser.parse_args(argv)
    text = generate()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print(f"{OUTPUT.relative_to(ROOT)} is stale: run {REGENERATE}")
            return 1
        return 0
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
