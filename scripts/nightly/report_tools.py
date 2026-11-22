"""Read-only views of the nightly's results, for the report agent.

The agent that writes a red night's `report.md` may run this script and
`git log/show/diff` on the nightly tree, and nothing else (`agent_command` in
nightly.py). Every subcommand only reads; none writes a file. In particular
`triage.triage()` writes `triage.json` into the night it reads, so nothing
here calls it.

    report_tools.py nights RESULTS
    report_tools.py summary NIGHT
    report_tools.py diff NIGHT BASE EXE DECK
    report_tools.py sweep RESULTS CASE
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import triage  # noqa: E402


def _json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def nights(results: Path) -> str:
    """One line per night: date, commit, cells, status, counts."""
    out = []
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        meta, tri = _json(d / "night.json"), _json(d / "triage.json")
        out.append(
            f"{d.name}  {str(meta.get('sha', ''))[:12]:12}  "
            f"{','.join(meta.get('cells', []) or []):40}  "
            f"{tri.get('status', '-'):5}  {json.dumps(tri.get('counts', {}))}"
        )
    return "\n".join(out)


def summary(night: Path, per_group: int = 5) -> str:
    """The night's findings grouped by (kind, severity, exe, cell), with the
    group size and the first few decks or cases of each."""
    tri = _json(night / "triage.json")
    if not tri:
        return f"{night}: no triage.json"
    groups = collections.defaultdict(list)
    for f in tri.get("findings", []):
        key = (f["kind"], f["severity"], f.get("exe") or "-", f.get("cell") or "-")
        groups[key].append(f)
    lines = [
        f"night {night.name}  status {tri.get('status')}  base {str(tri.get('base'))[:12]}"
        f" ({tri.get('base_date')})  counts {json.dumps(tri.get('counts', {}))}"
    ]
    for key, fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        details = collections.Counter(f["detail"] for f in fs)
        lines.append(
            f"\n[{len(fs)}] kind={key[0]} severity={key[1]} exe={key[2]} cell={key[3]}"
        )
        for d, n in details.most_common(3):
            lines.append(f"    detail x{n}: {d}")
        for f in fs[:per_group]:
            lines.append(f"    - {f.get('deck') or f.get('case')}")
        if len(fs) > per_group:
            lines.append(f"    ... {len(fs) - per_group} more")
    return "\n".join(lines)


def diff(night: Path, base: Path, exe: str, deck: str, context: int = 2) -> str:
    """Tonight's and the base's avx2-T1 Python printout of one deck, line 2
    (momwire's identity) masked as triage masks it."""
    rows, brows = triage.audit_rows(night), triage.audit_rows(base)
    cell = "avx2-T1"
    r = rows.get(cell, {}).get((exe, deck))
    b = brows.get(cell, {}).get((exe, deck))
    if r is None or b is None:
        return f"no {cell} row for ({exe}, {deck}) in {'tonight' if r is None else 'the base'}"
    got = triage.Printouts(night).get(r.get("key", ""))
    ref = triage.Printouts(base).get(b.get("key", ""))
    if got is None or ref is None:
        return "printout not saved for one of the nights"
    a = triage.without_identity(ref).decode("latin-1").splitlines()
    c = triage.without_identity(got).decode("latin-1").splitlines()
    out = list(
        difflib.unified_diff(
            a, c, f"base {base.name}", f"night {night.name}", n=context, lineterm=""
        )
    )
    return "\n".join(out[:200]) if out else "identical once line 2 is masked"


def sweep(results: Path, case: str) -> str:
    """One sweep case across every night: commit, wall time, Z, outcome."""
    out = []
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        p = d / "sweep.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("id") != case:
                continue
            wall = r.get("wall_s")
            out.append(
                f"{d.name}  {str(r.get('momwire_sha', ''))[:12]:12}  "
                f"wall {wall if wall is None else round(wall, 3)!s:>7} s  "
                f"z {r.get('z')}  {r.get('outcome')}  mode {r.get('mode')}  load1 {r.get('load1')}"
            )
    return "\n".join(out) or f"no rows for {case}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("nights").add_argument("results", type=Path)
    sub.add_parser("summary").add_argument("night", type=Path)
    d = sub.add_parser("diff")
    for name in ("night", "base"):
        d.add_argument(name, type=Path)
    d.add_argument("exe")
    d.add_argument("deck")
    s = sub.add_parser("sweep")
    s.add_argument("results", type=Path)
    s.add_argument("case")
    a = ap.parse_args(argv)
    if a.cmd == "nights":
        print(nights(a.results))
    elif a.cmd == "summary":
        print(summary(a.night))
    elif a.cmd == "diff":
        print(diff(a.night, a.base, a.exe, a.deck))
    else:
        print(sweep(a.results, a.case))
    return 0


if __name__ == "__main__":
    sys.exit(main())
