"""momwire#1064 G6, Population B: the census's momwire runner on main and on the
branch, compared deck by deck.

  python g6_popb_compare.py

Reads `popb_momwire_main.jsonl` and `popb_momwire_branch.jsonl`, the
`scratch/896-census/census_momwire.py` reports (a `_meta` header, then one
`{file, status, exit_code, wall_s, error, z}` row per deck). Per deck: the
status and the error sentence on both trees, and, where both served, the
largest relative change over every impedance the row carries. A deck present on
one tree only is a failure.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = 4.7e-4


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_meta" in r:
            continue
        rows[r["file"]] = r
    return rows


def impedances(z):
    """Every [re, im] pair in a row's `z`, in order, however it is nested."""
    out = []

    def walk(v):
        if (
            isinstance(v, (list, tuple))
            and len(v) == 2
            and all(isinstance(x, (int, float)) for x in v)
        ):
            out.append(complex(v[0], v[1]))
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(z)
    return out


def main():
    main_rows = load(HERE / "popb_momwire_main.jsonl")
    branch_rows = load(HERE / "popb_momwire_branch.jsonl")
    rows, fails = [], []
    for name in sorted(set(main_rows) | set(branch_rows)):
        rm, rb = main_rows.get(name), branch_rows.get(name)
        if rm is None or rb is None:
            fails.append(f"{name}: present on one tree only")
            continue
        row = dict(
            file=name,
            status_main=rm.get("status"),
            status_branch=rb.get("status"),
            same_status=rm.get("status") == rb.get("status"),
            same_error=(rm.get("error") or "") == (rb.get("error") or ""),
            error_main=(rm.get("error") or "")[:300],
        )
        zm, zb = impedances(rm.get("z")), impedances(rb.get("z"))
        if zm and zb:
            if len(zm) != len(zb):
                fails.append(f"{name}: the trees report different impedance counts")
            else:
                row["bitwise"] = zm == zb
                row["max_rel_z"] = max(
                    abs(b - a) / max(abs(a), 1e-300)
                    for a, b in zip(zm, zb, strict=True)
                )
        rows.append(row)
    out = dict(gate="G6 Population B", bar=BAR, failures=fails, rows=rows)
    (HERE / "g6_popb_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
